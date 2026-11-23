import agent_main


class FakePeTTa:
    def __init__(self):
        self.loaded_files = []
        self.expression = None

    def load_metta_file(self, path):
        self.loaded_files.append(path)

    def process_metta_string(self, expression):
        self.expression = expression
        return ["first line", "second line"]


def test_task_mode_prints_list_results_cleanly(monkeypatch, capsys):
    fake_petta = FakePeTTa()

    monkeypatch.setattr(agent_main, "PeTTa", lambda: fake_petta)
    monkeypatch.setattr(
        agent_main.agent_bridge, "init_runtime", lambda: 2
    )

    agent_main.main(["--task", "Test task"])

    output = capsys.readouterr().out.splitlines()

    assert output[-2:] == ["first line", "second line"]
    assert "['first line', 'second line']" not in output
    assert fake_petta.expression == '!(agent-run "Test task")'
