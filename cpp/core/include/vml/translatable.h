// Marks English text of the engine's that the window shows in the user's
// language (the window's tr(), cpp/app/i18n.h), so that
// scripts/i18n_catalog.py finds it among the catalogs' keys. The engine has
// no translations of its own: the text is returned as it is.
#pragma once

namespace vml {

constexpr const char* N_(const char* text) { return text; }

}  // namespace vml
