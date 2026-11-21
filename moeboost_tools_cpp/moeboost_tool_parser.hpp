#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include <iostream>
#include <regex>
#include <algorithm>
#include <cctype>

namespace moeboost {

struct ToolCall {
    std::string name;
    std::string arguments_raw;
    std::map<std::string, std::string> parameters;
};

struct ToolResult {
    std::string status;       // "success", "error", "confirmation_required"
    int exit_code = 0;
    std::string output;
    std::string extra_json;
};

class ToolParser {
public:
    static std::string stripThinking(const std::string& input) {
        // Strip <think>...</think> reasoning blocks cleanly
        static const std::regex think_regex(R"(<think\b[^>]*>[\s\S]*?</think\s*>)", std::regex::icase);
        return std::regex_replace(input, think_regex, "");
    }

    static std::vector<ToolCall> extractAllToolCalls(const std::string& reply_text) {
        std::vector<ToolCall> calls;
        std::string clean = stripThinking(reply_text);

        // 1) Qwen3-Coder / Colibri XML syntax:
        // <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        static const std::regex fn_regex(
            R"(<function\s*=\s*([a-zA-Z0-9_:-]+)\s*>([\s\S]*?)(?:</function\s*>|(?=</tool_call\s*>|$)))",
            std::regex::icase
        );

        auto fn_begin = std::sregex_iterator(clean.begin(), clean.end(), fn_regex);
        auto fn_end = std::sregex_iterator();

        for (auto it = fn_begin; it != fn_end; ++it) {
            std::smatch m = *it;
            ToolCall call;
            call.name = m[1].str();
            std::string body = m[2].str();

            static const std::regex param_regex(
                R"(<parameter\s*=\s*([a-zA-Z0-9_:-]+)\s*>([\s\S]*?)(?:</parameter\s*>|(?=<parameter\s*=|</function\s*>|</tool_call\s*>|$)))",
                std::regex::icase
            );
            auto p_begin = std::sregex_iterator(body.begin(), body.end(), param_regex);
            auto p_end = std::sregex_iterator();
            for (auto pit = p_begin; pit != p_end; ++pit) {
                std::smatch pm = *pit;
                std::string k = pm[1].str();
                std::string v = trim(pm[2].str());
                call.parameters[k] = v;
            }
            calls.push_back(call);
        }

        // 2) Standard attribute XML syntax: <function name="xxx"><parameter name="yyy">zzz</parameter></function>
        static const std::regex fn_attr_regex(
            R"(<function\s+name=["']([a-zA-Z0-9_:-]+)["'][^>]*>([\s\S]*?)</function\s*>)",
            std::regex::icase
        );
        auto fna_begin = std::sregex_iterator(clean.begin(), clean.end(), fn_attr_regex);
        for (auto it = fna_begin; it != fn_end; ++it) {
            std::smatch m = *it;
            ToolCall call;
            call.name = m[1].str();
            std::string body = m[2].str();

            static const std::regex pa_regex(
                R"(<parameter\s+name=["']([a-zA-Z0-9_:-]+)["'][^>]*>([\s\S]*?)</parameter\s*>)",
                std::regex::icase
            );
            auto pa_begin = std::sregex_iterator(body.begin(), body.end(), pa_regex);
            auto pa_end = std::sregex_iterator();
            for (auto pit = pa_begin; pit != pa_end; ++pit) {
                std::smatch pm = *pit;
                call.parameters[pm[1].str()] = trim(pm[2].str());
            }
            calls.push_back(call);
        }

        // 3) Fallback Markdown action blocks
        if (calls.empty()) {
            static const std::regex md_read(R"(```\s*read_file:([^`\n\r]+?)\s*```)", std::regex::icase);
            for (auto it = std::sregex_iterator(clean.begin(), clean.end(), md_read); it != fn_end; ++it) {
                std::smatch m = *it;
                ToolCall call;
                call.name = "read_workspace_file";
                call.parameters["path"] = trim(m[1].str());
                calls.push_back(call);
            }

            static const std::regex md_bash(R"(```(?:bash|sh)\s+([\s\S]*?)\s*```)", std::regex::icase);
            for (auto it = std::sregex_iterator(clean.begin(), clean.end(), md_bash); it != fn_end; ++it) {
                std::smatch m = *it;
                ToolCall call;
                call.name = "execute_terminal";
                call.parameters["command"] = trim(m[1].str());
                calls.push_back(call);
            }
        }

        return calls;
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    }
};

} // namespace moeboost
