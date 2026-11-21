#include "moeboost_tool_parser.hpp"
#include "moeboost_tool_engine.hpp"
#include <iostream>
#include <cassert>

using namespace moeboost;

int main() {
    std::cout << "=================================================================\n";
    std::cout << " MoEBoost C++: Coding, Editing, Patching, Debugging, Syntax Check\n";
    std::cout << "=================================================================\n";

    WorkspaceManager::instance().setWorkspaceDir("/home/local-ai");
    std::string test_file = "moeboost_tools_cpp/sample_code.py";

    // 1. CODING: Write initial code file
    std::string code =
        "# Sample module\n"
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "\n"
        "    def multiply(self, a, b):\n"
        "        return a * b\n";

    ToolResult res = ToolDispatcher::dispatch("write_file", {{"path", test_file}, {"content", code}});
    std::cout << "[1. Write Code] " << res.status << " - " << res.output << "\n";
    assert(res.status == "success");

    // 2. SYNTAX CHECK: Python syntax compilation
    res = ToolDispatcher::dispatch("check_syntax", {{"path", test_file}});
    std::cout << "[2. Check Syntax] " << res.status << " - " << (res.output.empty() ? "No syntax errors" : res.output) << "\n";
    assert(res.status == "success");

    // 3. CODE OUTLINE: Extract definitions (classes & defs)
    res = ToolDispatcher::dispatch("outline", {{"path", test_file}});
    std::cout << "[3. Code Outline]\n" << res.output;
    assert(res.status == "success");
    assert(res.output.find("class Calculator") != std::string::npos);
    assert(res.output.find("def add") != std::string::npos);

    // 4. SYMBOL SEARCH: Find definitions across files
    res = ToolDispatcher::dispatch("find_symbol", {{"symbol", "Calculator"}, {"path", "moeboost_tools_cpp"}});
    std::cout << "[4. Find Symbol]\n" << res.output;
    assert(res.status == "success");
    assert(res.output.find("Calculator") != std::string::npos);

    // 5. PATCHING: Replace method implementation
    std::string search_pattern = "    def multiply(self, a, b):\n        return a * b";
    std::string replacement    = "    def multiply(self, a, b):\n        # Fast multiply\n        return a * b";
    res = ToolDispatcher::dispatch("patch_file", {{"path", test_file}, {"search_pattern", search_pattern}, {"replacement", replacement}});
    std::cout << "[5. Patch File] " << res.status << " - " << res.output << "\n";
    assert(res.status == "success");

    // 6. EDIT LINES: Replace line range
    res = ToolDispatcher::dispatch("edit_lines", {{"path", test_file}, {"start_line", "1"}, {"end_line", "1"}, {"replacement", "# Upgraded Calculator Module"}});
    std::cout << "[6. Edit Lines] " << res.status << " - " << res.output << "\n";
    assert(res.status == "success");

    // Verify Read
    res = ToolDispatcher::dispatch("read_file", {{"path", test_file}, {"line_start", "1"}, {"line_count", "10"}});
    std::cout << "[7. Read File Content]\n" << res.output;
    assert(res.output.find("Upgraded Calculator Module") != std::string::npos);
    assert(res.output.find("Fast multiply") != std::string::npos);

    // 7. DEBUGGING & DIAGNOSTICS: Automated tool failure diagnostic
    res = ToolDispatcher::dispatch("diagnose_tool_error", {
        {"tool_name", "read_file"},
        {"target", "missing_secret.py"},
        {"error", "FileNotFoundError: [Errno 2] No such file or directory: 'missing_secret.py'"}
    });
    std::cout << "[8. Diagnose Error]\n" << res.output;
    assert(res.status == "success");
    assert(res.output.find("Path Resolution") != std::string::npos);

    // 8. PROJECT INTEGRITY CHECKS: Project-wide syntax validation
    res = ToolDispatcher::dispatch("run_project_checks", {{"path", "moeboost_tools_cpp"}});
    std::cout << "[9. Run Project Checks] " << res.status << " - " << res.output;
    assert(res.status == "success");

    std::cout << "\n=================================================================\n";
    std::cout << " ALL CODING, EDITING, PATCHING, DEBUGGING & SYNTAX TESTS PASSED!\n";
    std::cout << "=================================================================\n";
    return 0;
}
