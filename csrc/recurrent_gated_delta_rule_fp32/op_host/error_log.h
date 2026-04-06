#ifndef RECURRENT_GATED_DELTA_RULE_ERROR_LOG_H_
#define RECURRENT_GATED_DELTA_RULE_ERROR_LOG_H_

#include <string>
#include "toolchain/slog.h"

#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)             \
    do {                                 \
        printf("[WARN][%s] ", (opname)); \
        printf(__VA_ARGS__);             \
        printf("\n");                    \
    } while (0)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) \
    do {                                    \
        printf("[ERRORx][%s] ", (opname));  \
        printf(__VA_ARGS__);                \
        printf("\n");                       \
    } while (0)

#define OP_LOGE(opname, ...)              \
    do {                                  \
        printf("[ERROR][%s] ", (opname)); \
        printf(__VA_ARGS__);              \
        printf("\n");                     \
    } while (0)

#define OP_LOGD(opname, ...)

#define OPS_REPORT_CUBE_INNER_ERR(opname, ...) OP_LOGE(opname, __VA_ARGS__)
#define OPS_REPORT_VECTOR_INNER_ERR(opname, ...) OP_LOGE(opname, __VA_ARGS__)

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)   \
    do {                                                         \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
    } while (0)

#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                  \
        if (cond) {                       \
            log_func;                     \
            expr;                         \
        }                                 \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                     \
    do {                                                             \
        if ((ptr) == nullptr) {                                      \
            OP_LOGE(context->GetNodeType(), "%s is null", #ptr);     \
            return ge::GRAPH_FAILED;                                 \
        }                                                            \
    } while (0)

}  // namespace optiling

#endif  // RECURRENT_GATED_DELTA_RULE_ERROR_LOG_H_
