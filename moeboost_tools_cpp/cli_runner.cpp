#include "moeboost_tool_parser.hpp"
#include "moeboost_tool_engine.hpp"
#include <iostream>
#include <sstream>

using namespace moeboost;

int main(int argc, char* argv[]) {
    // Read input either from argv[1] or stdin
    std::string input;
    if (argc > 1) {
        input = argv[1];
    } else {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        input = ss.str();
    }

    if (input.empty()) {
        std::cerr << "Usage: moeboost_cli \"<tool_call>...\" or via stdin\n";
        return 1;
    }

    auto calls = ToolParser::extractAllToolCalls(input);
    if (calls.empty()) {
        std::cerr << "[moeboost_cli] No tool call detected in input.\n";
        return 2;
    }

    for (const auto& call : calls) {
        std::cout << "[moeboost_cli] Executing: " << call.name << "\n";
        ToolResult res = ToolDispatcher::dispatch(call.name, call.parameters);
        std::cout << "Status: " << res.status << " (exit code " << res.exit_code << ")\n";
        std::cout << "Output:\n" << res.output << "\n";
    }

    return 0;
}
