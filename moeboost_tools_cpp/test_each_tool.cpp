#include "moeboost_tool_parser.hpp"
#include "moeboost_tool_engine.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace moeboost;

void run_test(const std::string& tool_name, const std::string& desc, bool condition, const std::string& output = "") {
    std::cout << std::left << std::setw(28) << ("[" + tool_name + "]")
              << std::setw(36) << desc
              << " : " << (condition ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
              << "\n";
    if (!condition || !output.empty()) {
        std::cout << "    Detail: " << output << "\n";
    }
    assert(condition);
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << " MoEBoost C++: Comprehensive Test Suite for EVERY Registered Tool\n";
    std::cout << "======================================================================\n\n";

    WorkspaceManager::instance().setWorkspaceDir("/home/local-ai");
    std::string test_dir = "moeboost_tools_cpp/test_sandbox";
    std::string py_file = test_dir + "/sample.py";
    std::string cpp_file = test_dir + "/sample.cpp";
    std::string sh_file = test_dir + "/sample.sh";
    std::string json_file = test_dir + "/sample.json";

    // 1. ToolParser::stripThinking
    std::string thought_xml = "<think>Secret CoT</think><tool_call><function=test></function></tool_call>";
    std::string stripped = ToolParser::stripThinking(thought_xml);
    run_test("ToolParser::stripThinking", "Strips CoT think blocks",
             stripped.find("Secret CoT") == std::string::npos && stripped.find("<function=test>") != std::string::npos);

    // 2. ToolParser::extractAllToolCalls
    auto calls = ToolParser::extractAllToolCalls(thought_xml);
    run_test("ToolParser::extractCalls", "Extracts XML function calls",
             calls.size() == 1 && calls[0].name == "test");

    // 3. write_workspace_file (alias: write_file)
    std::string py_code =
        "# Module sample\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        return 42\n"
        "\n"
        "def helper():\n"
        "    return 'ok'\n";
    ToolResult res = ToolDispatcher::dispatch("write_file", {{"path", py_file}, {"content", py_code}});
    run_test("write_workspace_file", "Creates and writes file", res.status == "success");

    // 4. read_workspace_file (alias: read_file)
    res = ToolDispatcher::dispatch("read_file", {{"path", py_file}, {"line_start", "1"}, {"line_count", "4"}});
    run_test("read_workspace_file", "Reads bounded line range",
             res.status == "success" && res.output.find("class Worker") != std::string::npos);

    // 5. patch_workspace_file (alias: patch_file)
    std::string search = "    def run(self):\n        return 42";
    std::string repl = "    def run(self):\n        # patched logic\n        return 100";
    res = ToolDispatcher::dispatch("patch_file", {{"path", py_file}, {"search_pattern", search}, {"replacement", repl}});
    run_test("patch_workspace_file", "Exact block patch replacement", res.status == "success");

    // Verify patch applied
    res = ToolDispatcher::dispatch("read_file", {{"path", py_file}, {"line_start", "1"}, {"line_count", "10"}});
    run_test("verify_patch", "Verify patched content in file", res.output.find("patched logic") != std::string::npos);

    // 6. edit_workspace_lines (alias: edit_lines)
    res = ToolDispatcher::dispatch("edit_lines", {{"path", py_file}, {"start_line", "1"}, {"end_line", "1"}, {"replacement", "# Patched Header"}});
    run_test("edit_workspace_lines", "In-place line range replacement", res.status == "success");

    // 7. list_workspace_files (alias: list_files)
    res = ToolDispatcher::dispatch("list_files", {{"relative_path", test_dir}});
    run_test("list_workspace_files", "Lists workspace directory",
             res.status == "success" && res.output.find("sample.py") != std::string::npos);

    // 8. grep_workspace (alias: grep, ggrep, search_code)
    res = ToolDispatcher::dispatch("ggrep", {{"pattern", "patched logic"}, {"path", test_dir}});
    run_test("grep_workspace", "Regex / text pattern search",
             res.status == "success" && res.output.find("sample.py:4") != std::string::npos);

    // 9. find_workspace_files (alias: find_files)
    res = ToolDispatcher::dispatch("find_files", {{"name_pattern", "sample.py"}, {"path", test_dir}});
    run_test("find_workspace_files", "Discovers file by pattern",
             res.status == "success" && res.output.find("sample.py") != std::string::npos);

    // 10. outline_python_file (alias: outline, outline_file)
    res = ToolDispatcher::dispatch("outline", {{"path", py_file}});
    run_test("outline_python_file", "Extracts classes and functions",
             res.status == "success" && res.output.find("class Worker") != std::string::npos && res.output.find("def helper") != std::string::npos);

    // 11. find_symbol_definitions (alias: find_symbol)
    res = ToolDispatcher::dispatch("find_symbol", {{"symbol", "Worker"}, {"path", test_dir}});
    run_test("find_symbol_definitions", "Locates symbol definitions",
             res.status == "success" && res.output.find("class Worker") != std::string::npos);

    // 12. check_python_file (alias: check_syntax, check_python)
    // 12a: Python syntax pass
    res = ToolDispatcher::dispatch("check_syntax", {{"path", py_file}});
    run_test("check_syntax (Python)", "Validates valid Python syntax", res.status == "success");

    // 12b: C++ syntax check
    std::string cpp_code = "#include <iostream>\nint main() { return 0; }\n";
    ToolDispatcher::dispatch("write_file", {{"path", cpp_file}, {"content", cpp_code}});
    res = ToolDispatcher::dispatch("check_syntax", {{"path", cpp_file}});
    run_test("check_syntax (C++)", "Validates valid C++ syntax", res.status == "success");

    // 12c: Shell syntax check
    std::string sh_code = "#!/bin/bash\necho 'hello world'\n";
    ToolDispatcher::dispatch("write_file", {{"path", sh_file}, {"content", sh_code}});
    res = ToolDispatcher::dispatch("check_syntax", {{"path", sh_file}});
    run_test("check_syntax (Bash)", "Validates valid Bash script", res.status == "success");

    // 12d: JSON syntax check
    std::string json_code = "{\"service\": \"moeboost\", \"port\": 4133}\n";
    ToolDispatcher::dispatch("write_file", {{"path", json_file}, {"content", json_code}});
    res = ToolDispatcher::dispatch("check_syntax", {{"path", json_file}});
    run_test("check_syntax (JSON)", "Validates valid JSON file", res.status == "success");

    // 13. diagnose_tool_error (alias: diagnose_error)
    res = ToolDispatcher::dispatch("diagnose_error", {
        {"tool_name", "read_file"},
        {"target", "unknown.py"},
        {"error", "FileNotFoundError: No such file"}
    });
    run_test("diagnose_tool_error", "Provides diagnostic remedies",
             res.status == "success" && res.output.find("Path Resolution") != std::string::npos);

    // 14. run_project_checks (alias: check_project)
    res = ToolDispatcher::dispatch("check_project", {{"path", test_dir}});
    run_test("run_project_checks", "Scans directory for syntax",
             res.status == "success" && res.output.find("0 failed") != std::string::npos);

    // 15. execute_terminal (alias: terminal, bash, sh)
    res = ToolDispatcher::dispatch("terminal", {{"command", "echo 'Terminal execution test successful'"}});
    run_test("execute_terminal", "Runs shell command via pipe",
             res.status == "success" && res.output.find("Terminal execution test successful") != std::string::npos);

    // 16. git_status (alias: git)
    res = ToolDispatcher::dispatch("git_status", {{"path", "/home/local-ai/strata-moeboost"}});
    run_test("git_status", "Inspects git repository status", res.status == "success");

    // 17. git_diff
    res = ToolDispatcher::dispatch("git_diff", {{"path", "/home/local-ai/strata-moeboost"}});
    run_test("git_diff", "Executes git diff inspection", res.status == "success");

    // 18. check_port_open (alias: check_port)
    res = ToolDispatcher::dispatch("check_port", {{"host", "127.0.0.1"}, {"port", "4133"}});
    run_test("check_port_open", "Probes live Strata-MoEBoost server",
             res.status == "success" && res.output.find("OPEN") != std::string::npos);

    // 19. inspect_workspace_permissions (alias: permissions)
    res = ToolDispatcher::dispatch("permissions", {{"path", py_file}});
    run_test("inspect_permissions", "Inspects file uid, gid, mode",
             res.status == "success" && res.output.find("Owner:") != std::string::npos);

    // 20. get_system_telemetry (alias: telemetry)
    res = ToolDispatcher::dispatch("telemetry", {});
    run_test("get_system_telemetry", "Reads system host memory",
             res.status == "success" && res.output.find("RAM Total:") != std::string::npos);

    // 21. Security: Path Traversal Containment
    ToolResult bad_escape = NativeToolEngine::readWorkspaceFile("../../../etc/passwd");
    run_test("security_containment", "Rejects escape outside workspace",
             bad_escape.status == "error" && bad_escape.output.find("Access denied") != std::string::npos);

    std::cout << "\n======================================================================\n";
    std::cout << " \033[32mALL 21 NATIVE C++ TOOLS VERIFIED AND PASSED SUCCESSFULLY!\033[0m\n";
    std::cout << "======================================================================\n";
    return 0;
}
