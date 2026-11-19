// include/strata/artifact/model_family.hpp - model families supported by Strata-MoEBoost.
// Extended to support:
//   - Qwen4Exp (Qwen3.8-Flash-Next)
//   - Qwen35Moe / Qwen3Moe (Qwen3.6-35B-A3B, Qwen3-Coder-REAP-25B-A3B)
//   - DeepSeek2Moe / DeepSeek3Moe (DeepSeek-V2 / DeepSeek-V3 / DeepSeek-Coder-V2)
//   - MixtralMoe (Mistral/Mixtral 8x7B, 8x22B)
//   - DbrxMoe (DBRX 16x12B)
#pragma once

#include <string>

namespace strata {

enum class ModelFamily {
    Unknown = 0,
    Qwen4Exp,       ///< Qwen3.8-Flash-Next (qwen4exp)
    Qwen35Moe,      ///< Qwen3.6-35B-A3B / Qwen3-Coder (qwen35moe, qwen3moe)
    DeepSeekMoe,    ///< DeepSeek-V2/V3 / DeepSeek-Coder-V2 (deepseek2, deepseek3)
    MixtralMoe,     ///< Mixtral 8x7B, 8x22B (mixtral, mistralmoe)
    DbrxMoe,        ///< DBRX (dbrx)
};

inline ModelFamily model_family_from_arch(const std::string& arch) {
    if (arch == "qwen4exp") return ModelFamily::Qwen4Exp;
    if (arch == "qwen35moe" || arch == "qwen3moe" || arch == "qwen2moe") return ModelFamily::Qwen35Moe;
    if (arch == "deepseek2" || arch == "deepseek3" || arch == "deepseek" || arch == "dsv3" || arch == "dsv2") return ModelFamily::DeepSeekMoe;
    if (arch == "mixtral" || arch == "mistralmoe") return ModelFamily::MixtralMoe;
    if (arch == "dbrx") return ModelFamily::DbrxMoe;
    return ModelFamily::Unknown;
}

/// The GGUF architecture string (also the metadata key prefix: "<arch>.block_count").
inline const char* model_family_arch(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "qwen4exp";
    case ModelFamily::Qwen35Moe:
        return "qwen35moe";
    case ModelFamily::DeepSeekMoe:
        return "deepseek2";
    case ModelFamily::MixtralMoe:
        return "mixtral";
    case ModelFamily::DbrxMoe:
        return "dbrx";
    default:
        return "";
    }
}

/// The model's name for messages.
inline const char* model_family_title(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "Qwen3.8-Flash-Next";
    case ModelFamily::Qwen35Moe:
        return "Qwen3.6 / Qwen3-MoE";
    case ModelFamily::DeepSeekMoe:
        return "DeepSeek-V2/V3 / Coder-V2";
    case ModelFamily::MixtralMoe:
        return "Mixtral-8x7B/8x22B";
    case ModelFamily::DbrxMoe:
        return "DBRX-MoE";
    default:
        return "an unknown model";
    }
}

/// Whether Flash-Next's engine path runs the family.
inline bool model_family_runs(ModelFamily f) {
    return f == ModelFamily::Qwen4Exp || f == ModelFamily::Qwen35Moe || 
           f == ModelFamily::DeepSeekMoe || f == ModelFamily::MixtralMoe || f == ModelFamily::DbrxMoe;
}

/// The refusal for a recognised family the engine does not run (empty when supported).
inline std::string model_family_not_supported(ModelFamily f) {
    if (model_family_runs(f)) return {};
    if (f == ModelFamily::Unknown) return "this is not a model family Strata-MoEBoost knows";
    return std::string(model_family_title(f)) + " (architecture '" + model_family_arch(f) +
           "') is not yet supported on this hardware configuration.";
}

}  // namespace strata
