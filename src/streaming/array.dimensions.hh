#pragma once

#include "zarr.types.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct ZarrDimension
{
    ZarrDimension() = default;
    ZarrDimension(std::string_view name,
                  ZarrDimensionType type,
                  uint32_t array_size_px,
                  uint32_t chunk_size_px,
                  uint32_t shard_size_chunks,
                  std::string_view unit = "",
                  double scale = 1.0)
      : name(name)
      , type(type)
      , array_size_px(array_size_px)
      , chunk_size_px(chunk_size_px)
      , shard_size_chunks(shard_size_chunks)
      , scale(scale)
    {
        if (!unit.empty()) {
            this->unit = unit;
        }
    }

    std::string name;
    ZarrDimensionType type{ ZarrDimensionType_Space };

    std::optional<std::string> unit;
    double scale{ 1.0 };

    uint32_t array_size_px{ 0 };
    uint32_t chunk_size_px{ 0 };
    uint32_t shard_size_chunks{ 0 };
};

class ArrayDimensions
{
  public:
    ArrayDimensions(std::vector<ZarrDimension>&& dims,
                    ZarrDataType dtype,
                    const std::vector<std::string>& target_dim_order = {});

    size_t ndims() const;

    const ZarrDimension& operator[](size_t idx) const;
    const ZarrDimension& at(size_t idx) const { return operator[](idx); }

    const ZarrDimension& final_dim() const;
    const ZarrDimension& height_dim() const;
    const ZarrDimension& width_dim() const;

    /**
     * @brief Get the dimension at the given index in canonical OME-NGFF order
     * (TCZYX).
     * @param idx The index in canonical order.
     * @return The dimension at the given index.
     */
    const ZarrDimension& canonical_dimension(size_t idx) const;

    /**
     * @brief Check if dimensions need transposition for OME-NGFF compliance.
     * @return True if dimensions are not in canonical TCZYX order.
     */
    bool needs_transposition() const;

    /**
     * @brief Get the index of a chunk in the chunk lattice for a given frame
     * and dimension.
     * @param frame_id The frame ID.
     * @param dimension_idx The index of the dimension in the dimension vector.
     * @return The index of the chunk in the chunk lattice.
     */
    uint32_t chunk_lattice_index(uint64_t frame_id, uint32_t dim_index) const;

    /**
     * @brief Find the offset in the array of chunk buffers for the given frame.
     * @param frame_id The frame ID.
     * @return The offset in the array of chunk buffers.
     */
    uint32_t tile_group_offset(uint64_t frame_id) const;

    /**
     * @brief Find the byte offset inside a chunk for a given frame and data
     * type.
     * @param frame_id The frame ID.
     * @param dims The dimensions of the array.
     * @param type The data type of the array.
     * @return The byte offset inside a chunk.
     */
    uint64_t chunk_internal_offset(uint64_t frame_id) const;

    /**
     * @brief Get the number of chunks to hold in memory.
     * @return The number of chunks to buffer before writing out.
     */
    uint32_t number_of_chunks_in_memory() const;

    /**
     * @brief Get the size, in bytes, of a single raw chunk.
     * @return The number of bytes to allocate for a chunk.
     */
    size_t bytes_per_chunk() const;

    /**
     * @brief Get the number of shards to write at one time.
     * @return The number of shards to buffer and write out.
     */
    uint32_t number_of_shards() const;

    /**
     * @brief Get the number of chunks in a single shard.
     * @return The number of chunks in a shard.
     */
    uint32_t chunks_per_shard() const;

    /**
     * @brief Get the number of chunk layers in a single shard.
     * @note The number of chunks per shard is the product of the number of
     * chunks in memory and the number of layers per shard.
     * @return The number of layers in a shard.
     */
    uint32_t chunk_layers_per_shard() const;

    /**
     * @brief Get the shard index for a given chunk index, given array dimensions.
     * @param chunk_index The index of the chunk.
     * @return The index of the shard containing the chunk.
     */
    uint32_t shard_index_for_chunk(uint32_t chunk_index) const;

    /**
     * @brief Get the chunk indices corresponding to a given shard index.
     * @param shard_index The index of the shard.
     * @return A vector of chunk indices corresponding to the shard.
     */
    const std::vector<uint32_t>& chunk_indices_for_shard(
      uint32_t shard_index) const;

    /**
     * @brief Get the chunk indices for a specific layer within a shard.
     * @param shard_index The index of the shard.
     * @param layer
     * @return
     */
    std::vector<uint32_t> chunk_indices_for_shard_layer(uint32_t shard_index,
                                                        uint32_t layer) const;

    /**
     * @brief Get the streaming index of a chunk within a shard.
     * @param chunk_index The index of the chunk.
     * @return The index of the chunk within the shard.
     */
    uint32_t shard_internal_index(uint32_t chunk_index) const;

    /**
     * @brief Transpose a frame ID from acquisition order to canonical OME-NGFF
     * order.
     *
     * Frame IDs are linear indices that encode only non-spatial dimensions
     * (e.g., Time, Channel, Z-depth). The spatial dimensions (Y, X) are always
     * implicit zeros in the frame_id coordinate space, as frame_id represents
     * which 2D image plane is being written, not individual pixels.
     *
     * When dimensions are provided in acquisition order (e.g., T, Z, C, Y, X),
     * frames are written sequentially with frame_id incrementing according to
     * that order. This function converts the acquisition frame_id to a
     * canonical frame_id that corresponds to OME-NGFF dimension order
     * (T, C, Z, Y, X).
     *
     * Example:
     *   Acquisition order: T(size=2), Z(size=3), C(size=2), Y, X
     *   Frame 0: T=0, Z=0, C=0 -> acquisition frame_id=0
     *   Frame 1: T=0, Z=0, C=1 -> acquisition frame_id=1
     *   Frame 2: T=0, Z=1, C=0 -> acquisition frame_id=2
     *
     *   Canonical order: T(size=2), C(size=2), Z(size=3), Y, X
     *   The same physical frames map to:
     *   Frame T=0,C=0,Z=0 -> canonical frame_id=0
     *   Frame T=0,C=1,Z=0 -> canonical frame_id=3
     *   Frame T=0,C=0,Z=1 -> canonical frame_id=1
     *
     *   So transpose_frame_id(1) = 3, transpose_frame_id(2) = 1, etc.
     *
     * @param frame_id Sequential frame counter in acquisition order.
     * @return Sequential frame counter in canonical TCZYX order.
     */
    uint64_t transpose_frame_id(uint64_t frame_id) const;

  private:
    // Transposition state - only allocated when dimension reordering is needed
    struct TranspositionState {
        std::vector<ZarrDimension> acquisition_dims;  // Original acquisition order
        std::vector<size_t> acq_to_canonical;         // Maps acq index -> canonical index
        std::vector<size_t> canonical_to_acq;         // Maps canonical index -> acq index
    };

    std::vector<ZarrDimension> dims_; // Dimensions in storage order
    std::unique_ptr<TranspositionState> transpose_state_;  // nullptr when no transposition

    ZarrDataType dtype_;

    size_t bytes_per_chunk_;

    uint32_t number_of_chunks_in_memory_;
    uint32_t chunks_per_shard_;
    uint32_t number_of_shards_;

    std::unordered_map<uint32_t, uint32_t> shard_indices_;
    std::unordered_map<uint32_t, uint32_t> shard_internal_indices_;
    std::vector<std::vector<uint32_t>> chunk_indices_for_shard_;

    uint32_t shard_index_for_chunk_(uint32_t chunk_index) const;
    uint32_t shard_internal_index_(uint32_t chunk_index) const;
};

using DimensionPartsFun = std::function<size_t(const ZarrDimension&)>;