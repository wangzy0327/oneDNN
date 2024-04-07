#include "gpu/amd/miopen_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_batch_normalization_fwd_t::execute(const exec_ctx_t &ctx) const {
    return miopen_batch_normalization_common_t::execute(
            ctx, ctx.stream()->engine(), pd());
}

status_t miopen_batch_normalization_bwd_t::execute(const exec_ctx_t &ctx) const {
    return miopen_batch_normalization_common_t::execute(
            ctx, ctx.stream()->engine(), pd());
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
