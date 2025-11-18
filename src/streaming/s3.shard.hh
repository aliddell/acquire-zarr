#pragma once

#include "shard.hh"

namespace zarr {
class S3Shard : public Shard {
public:
protected:
bool compress_and_flush_chunks_() override;
};
} // namespace zarr