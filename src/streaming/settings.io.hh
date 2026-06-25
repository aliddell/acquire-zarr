#pragma once

#include "acquire.zarr.h" // ZarrStreamSettings

#include <nlohmann/json.hpp>

#include <string>

// Load and dump ZarrStreamSettings from YAML/JSON config files. JSON is a
// subset of YAML, so a single parse path handles both.
namespace zarr {
// Parse a YAML (or JSON) document into JSON. Throws on malformed input.
nlohmann::json
config_text_to_json(const std::string& text);

// Emit a JSON document as YAML text.
std::string
json_to_yaml(const nlohmann::json& doc);

// Build owning stream settings from a config document. Allocates all arrays,
// dimensions, plates, and strings; free with destroy_loaded_settings. Throws
// std::exception with a descriptive message on malformed input.
void
json_to_settings(const nlohmann::json& doc, ZarrStreamSettings* out);

// Serialize stream settings into a config document.
nlohmann::json
settings_to_json(const ZarrStreamSettings* settings);

// Free everything json_to_settings allocated.
void
destroy_loaded_settings(ZarrStreamSettings* settings);
} // namespace zarr
