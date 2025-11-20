#include "array.dimensions.hh"
#include "macros.hh"
#include "zarr.common.hh"

#include <algorithm>
#include <array>
#include <span>
#include <tuple>
#include <unordered_map>

std::pair<std::vector<ZarrDimension>,
          std::optional<ArrayDimensions::TranspositionMap>>
ArrayDimensions::compute_transposition(
  std::vector<ZarrDimension>&& acquisition_dims,
  const std::vector<std::string>& target_dim_order)
{
    if (target_dim_order.empty()) {
        return { std::move(acquisition_dims), std::nullopt };
    }

    const auto n = acquisition_dims.size();
    TranspositionMap map;
    map.acquisition_dims = std::move(acquisition_dims);

    // Validate target order
    EXPECT(target_dim_order.size() == n,
           "Target dimension order must have ",
           n,
           " elements to match dimension count, got ",
           target_dim_order.size());

    // Validate that dimension 0 is not transposed away
    EXPECT(target_dim_order[0] == map.acquisition_dims[0].name,
           "Transposing dimension 0 ('",
           map.acquisition_dims[0].name,
           "') away from position 0 is not currently supported. "
           "The first dimension must remain first in storage_dimension_order.");

    // Build index mapping
    map.acq_to_storage.resize(n);
    map.storage_to_acq.resize(n);

    std::vector<ZarrDimension> storage_dims(n);
    for (size_t target_idx = 0; target_idx < n; ++target_idx) {
        const auto& target_name = target_dim_order[target_idx];

        // Find this name in acquisition dims
        bool found = false;
        for (size_t acq_idx = 0; acq_idx < n; ++acq_idx) {
            if (map.acquisition_dims[acq_idx].name == target_name) {
                storage_dims[target_idx] = map.acquisition_dims[acq_idx];
                map.acq_to_storage[acq_idx] = target_idx;
                map.storage_to_acq[target_idx] = acq_idx;
                found = true;
                break;
            }
        }

        EXPECT(found,
               "Dimension name '",
               target_name,
               "' in target order not found in dimensions array");
    }

    // Validate the reordered dimensions have spatial dims at the end
    EXPECT(storage_dims[n - 2].type == ZarrDimensionType_Space,
           "After reordering, second-to-last dimension must be spatial");
    EXPECT(storage_dims[n - 1].type == ZarrDimensionType_Space,
           "After reordering, last dimension must be spatial");

    // Check if transposition is actually needed (might be identity)
    bool is_identity = true;
    for (size_t i = 0; i < n; ++i) {
        if (map.acq_to_storage[i] != i) {
            is_identity = false;
            break;
        }
    }

    if (is_identity) {
        return { std::move(storage_dims), std::nullopt };
    }

    return { std::move(storage_dims), std::move(map) };
}

ArrayDimensions::ArrayDimensions(
  std::vector<ZarrDimension>&& dims,
  ZarrDataType dtype,
  const std::vector<std::string>& target_dim_order)
  : dtype_(dtype)
  , chunks_per_shard_(1)
  , number_of_shards_(1)
  , bytes_per_chunk_(zarr::bytes_of_type(dtype))
  , number_of_chunks_in_memory_(1)
{
    EXPECT(dims.size() > 2, "Array must have at least three dimensions.");

    const auto n = dims.size();

    // Validate that last 2 dimensions are spatial (Y, X)
    EXPECT(dims[n - 2].type == ZarrDimensionType_Space,
           "Second-to-last dimension must be spatial (Y axis), got type ",
           static_cast<int>(dims[n - 2].type));
    EXPECT(dims[n - 1].type == ZarrDimensionType_Space,
           "Last dimension must be spatial (X axis), got type ",
           static_cast<int>(dims[n - 1].type));

    std::tie(dims_, transpose_map_) =
      compute_transposition(std::move(dims), target_dim_order);

    // Now compute chunk/shard info using dimensions in storage order
    for (auto i = 0; i < dims_.size(); ++i) {
        const auto& dim = dims_[i];
        bytes_per_chunk_ *= dim.chunk_size_px;
        chunks_per_shard_ *= dim.shard_size_chunks;

        if (i > 0) {
            number_of_chunks_in_memory_ *= zarr::chunks_along_dimension(dim);
            number_of_shards_ *= zarr::shards_along_dimension(dim);
        }
    }

    chunk_indices_for_shard_.resize(number_of_shards_);

    for (auto i = 0; i < chunks_per_shard_ * number_of_shards_; ++i) {
        const auto shard_index = shard_index_for_chunk_(i);
        shard_indices_.insert_or_assign(i, shard_index);
        shard_internal_indices_.insert_or_assign(i, shard_internal_index_(i));

        chunk_indices_for_shard_[shard_index].push_back(i);
    }
}

size_t
ArrayDimensions::ndims() const
{
    return dims_.size();
}

const ZarrDimension&
ArrayDimensions::operator[](size_t idx) const
{
    return dims_[idx];
}

const ZarrDimension&
ArrayDimensions::final_dim() const
{
    return dims_[0];
}

const ZarrDimension&
ArrayDimensions::height_dim() const
{
    return dims_[ndims() - 2];
}

const ZarrDimension&
ArrayDimensions::width_dim() const
{
    return dims_.back();
}

uint32_t
ArrayDimensions::chunk_lattice_index(uint64_t frame_id,
                                     uint32_t dim_index) const
{
    // the last two dimensions are special cases
    EXPECT(dim_index < ndims() - 2, "Invalid dimension index: ", dim_index);

    // the first dimension is a special case
    if (dim_index == 0) {
        auto divisor = dims_.front().chunk_size_px;
        for (auto i = 1; i < ndims() - 2; ++i) {
            const auto& dim = dims_[i];
            divisor *= dim.array_size_px;
        }

        CHECK(divisor);
        return frame_id / divisor;
    }

    size_t mod_divisor = 1, div_divisor = 1;
    for (auto i = dim_index; i < ndims() - 2; ++i) {
        const auto& dim = dims_[i];
        mod_divisor *= dim.array_size_px;
        div_divisor *= (i == dim_index ? dim.chunk_size_px : dim.array_size_px);
    }

    CHECK(mod_divisor);
    CHECK(div_divisor);

    return (frame_id % mod_divisor) / div_divisor;
}

uint32_t
ArrayDimensions::tile_group_offset(uint64_t frame_id) const
{
    std::vector<size_t> strides(dims_.size(), 1);
    for (auto i = dims_.size() - 1; i > 0; --i) {
        const auto& dim = dims_[i];
        const auto a = dim.array_size_px, c = dim.chunk_size_px;
        strides[i - 1] = strides[i] * ((a + c - 1) / c);
    }

    size_t offset = 0;
    for (auto i = ndims() - 3; i > 0; --i) {
        const auto idx = chunk_lattice_index(frame_id, i);
        const auto stride = strides[i];
        offset += idx * stride;
    }

    return offset;
}

uint64_t
ArrayDimensions::chunk_internal_offset(uint64_t frame_id) const
{
    const auto tile_size = zarr::bytes_of_type(dtype_) *
                           width_dim().chunk_size_px *
                           height_dim().chunk_size_px;

    uint64_t offset = 0;
    std::vector<uint64_t> array_strides(ndims() - 2, 1),
      chunk_strides(ndims() - 2, 1);

    for (auto i = (int)ndims() - 3; i > 0; --i) {
        const auto& dim = dims_[i];
        const auto internal_idx =
          (frame_id / array_strides[i]) % dim.array_size_px % dim.chunk_size_px;

        array_strides[i - 1] = array_strides[i] * dim.array_size_px;
        chunk_strides[i - 1] = chunk_strides[i] * dim.chunk_size_px;
        offset += internal_idx * chunk_strides[i];
    }

    // final dimension
    {
        const auto& dim = dims_[0];
        const auto internal_idx =
          (frame_id / array_strides.front()) % dim.chunk_size_px;
        offset += internal_idx * chunk_strides.front();
    }

    return offset * tile_size;
}

uint32_t
ArrayDimensions::number_of_chunks_in_memory() const
{
    return number_of_chunks_in_memory_;
}

size_t
ArrayDimensions::bytes_per_chunk() const
{
    return bytes_per_chunk_;
}

uint32_t
ArrayDimensions::number_of_shards() const
{
    return number_of_shards_;
}

uint32_t
ArrayDimensions::chunks_per_shard() const
{
    return chunks_per_shard_;
}

uint32_t
ArrayDimensions::chunk_layers_per_shard() const
{
    return dims_[0].shard_size_chunks;
}

uint32_t
ArrayDimensions::shard_index_for_chunk(uint32_t chunk_index) const
{
    return shard_indices_.at(chunk_index);
}

const std::vector<uint32_t>&
ArrayDimensions::chunk_indices_for_shard(uint32_t shard_index) const
{
    return chunk_indices_for_shard_.at(shard_index);
}

std::vector<uint32_t>
ArrayDimensions::chunk_indices_for_shard_layer(uint32_t shard_index,
                                               uint32_t layer) const
{
    const auto& chunk_indices = chunk_indices_for_shard(shard_index);
    const auto chunks_per_layer = number_of_chunks_in_memory_;

    std::vector<uint32_t> indices;
    indices.reserve(chunks_per_shard_);

    for (const auto& idx : chunk_indices) {
        if ((idx / chunks_per_layer) == layer) {
            indices.push_back(idx);
        }
    }

    return indices;
}

uint32_t
ArrayDimensions::shard_internal_index(uint32_t chunk_index) const
{
    return shard_internal_indices_.at(chunk_index);
}

uint32_t
ArrayDimensions::shard_index_for_chunk_(uint32_t chunk_index) const
{
    // make chunk strides
    std::vector<uint64_t> chunk_strides;
    chunk_strides.resize(dims_.size());
    chunk_strides.back() = 1;

    for (auto i = dims_.size() - 1; i > 0; --i) {
        const auto& dim = dims_[i];
        chunk_strides[i - 1] =
          chunk_strides[i] * zarr::chunks_along_dimension(dim);
    }

    // get chunk indices
    std::vector<uint32_t> chunk_lattice_indices(ndims());
    for (auto i = ndims() - 1; i > 0; --i) {
        chunk_lattice_indices[i] =
          chunk_index % chunk_strides[i - 1] / chunk_strides[i];
    }

    // make shard strides
    std::vector<uint32_t> shard_strides(ndims(), 1);
    for (auto i = ndims() - 1; i > 0; --i) {
        const auto& dim = dims_[i];
        shard_strides[i - 1] =
          shard_strides[i] * zarr::shards_along_dimension(dim);
    }

    std::vector<uint32_t> shard_lattice_indices;
    for (auto i = 0; i < ndims(); ++i) {
        shard_lattice_indices.push_back(chunk_lattice_indices[i] /
                                        dims_[i].shard_size_chunks);
    }

    uint32_t index = 0;
    for (auto i = 0; i < ndims(); ++i) {
        index += shard_lattice_indices[i] * shard_strides[i];
    }

    return index;
}

uint32_t
ArrayDimensions::shard_internal_index_(uint32_t chunk_index) const
{
    // make chunk strides
    std::vector<uint64_t> chunk_strides;
    chunk_strides.resize(dims_.size());
    chunk_strides.back() = 1;

    for (auto i = dims_.size() - 1; i > 0; --i) {
        const auto& dim = dims_[i];
        chunk_strides[i - 1] =
          chunk_strides[i] * zarr::chunks_along_dimension(dim);
    }

    // get chunk indices
    std::vector<size_t> chunk_lattice_indices(ndims());
    for (auto i = ndims() - 1; i > 0; --i) {
        chunk_lattice_indices[i] =
          chunk_index % chunk_strides[i - 1] / chunk_strides[i];
    }
    chunk_lattice_indices[0] = chunk_index / chunk_strides.front();

    // make shard lattice indices
    std::vector<size_t> shard_lattice_indices;
    for (auto i = 0; i < ndims(); ++i) {
        shard_lattice_indices.push_back(chunk_lattice_indices[i] /
                                        dims_[i].shard_size_chunks);
    }

    std::vector<size_t> chunk_internal_strides(ndims(), 1);
    for (auto i = ndims() - 1; i > 0; --i) {
        const auto& dim = dims_[i];
        chunk_internal_strides[i - 1] =
          chunk_internal_strides[i] * dim.shard_size_chunks;
    }

    size_t index = 0;

    for (auto i = 0; i < ndims(); ++i) {
        index += (chunk_lattice_indices[i] % dims_[i].shard_size_chunks) *
                 chunk_internal_strides[i];
    }

    return index;
}

const ZarrDimension&
ArrayDimensions::storage_dimension(size_t idx) const
{
    return dims_[idx];
}

bool
ArrayDimensions::needs_transposition() const
{
    return transpose_map_.has_value();
}

bool
ArrayDimensions::needs_spatial_transposition() const
{
    if (!transpose_map_) {
        return false;
    }

    const auto n = ndims();
    // Check if the last two spatial dimensions (height and width) are swapped.
    // If acq[n-2] maps to storage_order[n-1] and acq[n-1] maps to storage_order[n-2],
    // then height and width are swapped (Y↔X).
    return transpose_map_->acq_to_storage[n - 2] == n - 1 &&
           transpose_map_->acq_to_storage[n - 1] == n - 2;
}

uint32_t
ArrayDimensions::acquisition_frame_rows() const
{
    const auto n = ndims();
    if (!transpose_map_) {
        // No transposition, acquisition order = storage order
        return dims_[n - 2].array_size_px;
    }
    // Return height from acquisition dimensions
    return transpose_map_->acquisition_dims[n - 2].array_size_px;
}

uint32_t
ArrayDimensions::acquisition_frame_cols() const
{
    const auto n = ndims();
    if (!transpose_map_) {
        // No transposition, acquisition order = storage order
        return dims_[n - 1].array_size_px;
    }
    // Return width from acquisition dimensions
    return transpose_map_->acquisition_dims[n - 1].array_size_px;
}

// Transpose a frame ID from acquisition order to output storage_dimension_order
uint64_t
ArrayDimensions::transpose_frame_id(uint64_t frame_id) const
{
    // Fast path: no transposition needed
    if (!transpose_map_) {
        return frame_id;
    }

    // NOTE:
    // We could potentially pre-compute a lookup table for frame_id
    // transposition, but only for fixed-size dimensions. (Append dimensions
    // would still need on-the-fly computation?).  Opted for simpler on-the-fly
    // computation for now.

    const auto n = ndims();

    // Use stack-allocated arrays for common case (most acquisitions have 3-7
    // dimensions). This avoids heap allocations on every frame write.
    constexpr size_t kMaxStackDims = 8;
    std::array<uint64_t, kMaxStackDims> acq_coords_stack{};
    std::array<uint64_t, kMaxStackDims> stor_coords_stack{};
    std::array<uint64_t, kMaxStackDims> acq_strides_stack{};
    std::array<uint64_t, kMaxStackDims> stor_strides_stack{};

    // Fallback to heap allocation for unusual cases with many dimensions
    std::vector<uint64_t> acq_coords_heap;
    std::vector<uint64_t> stor_coords_heap;
    std::vector<uint64_t> acq_strides_heap;
    std::vector<uint64_t> stor_strides_heap;

    std::span<uint64_t> acq_coords(acq_coords_stack);
    std::span<uint64_t> stor_coords(stor_coords_stack);
    std::span<uint64_t> acq_strides(acq_strides_stack);
    std::span<uint64_t> stor_strides(stor_strides_stack);

    if (n > kMaxStackDims) {
        acq_coords_heap.resize(n);
        stor_coords_heap.resize(n);
        acq_strides_heap.resize(n);
        stor_strides_heap.resize(n);
        acq_coords = std::span(acq_coords_heap);
        stor_coords = std::span(stor_coords_heap);
        acq_strides = std::span(acq_strides_heap);
        stor_strides = std::span(stor_strides_heap);
    } else {
        acq_coords = acq_coords.first(n);
        stor_coords = stor_coords.first(n);
        acq_strides = acq_strides.first(n);
        stor_strides = stor_strides.first(n);
    }

    std::ranges::fill(acq_strides, 1);
    std::ranges::fill(stor_strides, 1);

    // Step 1: Calculate strides in acquisition order for every frame-addressable
    // axis (all dimensions except the trailing plane axes). frame_id enumerates
    // only dims[0..n-3]; the final two spatial dimensions (typically Y, X) stay
    // at zero in this coordinate space.
    if (n > 2) {
        acq_strides[n - 3] = 1;
        for (int i = static_cast<int>(n) - 4; i >= 0; --i) {
            acq_strides[i] =
              acq_strides[i + 1] *
              transpose_map_->acquisition_dims[i + 1].array_size_px;
        }
    }

    // Step 2: Convert linear frame_id to multi-dimensional coordinates in
    // acquisition order
    uint64_t remaining = frame_id;
    for (size_t i = 0; i < n - 2; ++i) {
        acq_coords[i] = remaining / acq_strides[i];
        remaining %= acq_strides[i];
    }
    // Spatial dimensions (last 2) are always 0 in frame_id space
    acq_coords[n - 2] = 0;
    acq_coords[n - 1] = 0;

    // Step 3: Permute coordinates from acquisition order to storage order
    for (size_t i = 0; i < n; ++i) {
        stor_coords[transpose_map_->acq_to_storage[i]] = acq_coords[i];
    }

    // Step 4: Calculate strides in storage dimension order
    if (n > 2) {
        stor_strides[n - 3] = 1;
        for (int i = static_cast<int>(n) - 4; i >= 0; --i) {
            stor_strides[i] = stor_strides[i + 1] * dims_[i + 1].array_size_px;
        }
    }

    // Step 5: Convert storage dimension coordinates back to linear frame_id
    uint64_t storage_frame_id = 0;
    for (size_t i = 0; i < n - 2; ++i) {
        storage_frame_id += stor_coords[i] * stor_strides[i];
    }

    return storage_frame_id;
}
