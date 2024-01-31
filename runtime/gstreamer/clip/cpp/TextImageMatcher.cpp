#include "TextImageMatcher.hpp"

// Define static members
std::unordered_map<std::string, TextImageMatcher*> TextImageMatcher::instances;
std::mutex TextImageMatcher::mutex;

TextImageMatcher* TextImageMatcher::getInstance(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = instances.find(id);
    if (it != instances.end()) {
        // Instance already exists
        return it->second;
    } else {
        // Create new instance
        TextImageMatcher* instance = new TextImageMatcher(id);
        instances[id] = instance;
        return instance;
    }
}
