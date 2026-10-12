#ifndef DEMO_INCLUDE_YAML_UTIL_H_
#define DEMO_INCLUDE_YAML_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "demo/define.h"

std::string ResolveCoreAllocationPath(const std::string& ctor_override);

std::map<std::string, std::vector<CoreId>> LoadVisionCoreAllocation(
    const std::string& path);

#endif
