"""storage: locked appends, record formatting, IDs, and save verification."""

from support_crew import storage


def test_append_creates_file_and_returns_record_id(tmp_path):
    target = tmp_path / "answers.txt"
    record_id = storage.append_record("q", "a1", "a2", path=target)
    assert target.exists()
    assert len(record_id) == 32  # uuid4 hex


def test_record_contains_query_answers_and_id(tmp_path):
    target = tmp_path / "answers.txt"
    record_id = storage.append_record(
        "How do I log in?", "Direct answer.", "Web answer.", path=target
    )
    text = target.read_text(encoding="utf-8")
    assert f"Record-ID: {record_id}" in text
    assert "Query:\nHow do I log in?" in text
    assert "Assistant Answer:\nDirect answer." in text
    assert "Web Search Answer:\nWeb answer." in text


def test_append_preserves_earlier_records(tmp_path):
    target = tmp_path / "answers.txt"
    first = storage.append_record("q1", "a", "b", path=target)
    second = storage.append_record("q2", "c", "d", path=target)
    text = target.read_text(encoding="utf-8")
    assert f"Record-ID: {first}" in text
    assert f"Record-ID: {second}" in text
    assert first != second


def test_append_strips_surrounding_whitespace(tmp_path):
    target = tmp_path / "answers.txt"
    storage.append_record("  q  \n", "\n a1 ", " a2 ", path=target)
    text = target.read_text(encoding="utf-8")
    assert "Query:\nq\n" in text
    assert "Assistant Answer:\na1\n" in text


def test_append_handles_unicode(tmp_path):
    target = tmp_path / "answers.txt"
    storage.append_record("¿Cómo inicio sesión? 🤝", "Réponse", "答案", path=target)
    text = target.read_text(encoding="utf-8")
    assert "¿Cómo inicio sesión? 🤝" in text
    assert "答案" in text


def test_last_record_id_tracks_current_thread_append(tmp_path):
    target = tmp_path / "answers.txt"
    storage.reset_last_record_id()
    assert storage.get_last_record_id() is None
    record_id = storage.append_record("q", "a", "b", path=target)
    assert storage.get_last_record_id() == record_id


def test_record_exists_finds_only_real_ids(tmp_path):
    target = tmp_path / "answers.txt"
    record_id = storage.append_record("q", "a", "b", path=target)
    assert storage.record_exists(record_id, path=target) is True
    assert storage.record_exists("0" * 32, path=target) is False
    assert storage.record_exists(None, path=target) is False
    assert storage.record_exists(record_id, path=tmp_path / "missing.txt") is False


def test_rotation_archives_file_once_cap_is_reached(tmp_path):
    target = tmp_path / "answers.txt"
    first = storage.append_record("q1", "a", "b", path=target, max_bytes=100)
    # File now exceeds the 100-byte cap, so the next append rotates first.
    second = storage.append_record("q2", "c", "d", path=target, max_bytes=100)

    archives = sorted(tmp_path.glob("answers-*.txt"))
    assert len(archives) == 1
    assert f"Record-ID: {first}" in archives[0].read_text(encoding="utf-8")

    current = target.read_text(encoding="utf-8")
    assert f"Record-ID: {second}" in current
    assert f"Record-ID: {first}" not in current


def test_rotation_disabled_with_zero_cap(tmp_path):
    target = tmp_path / "answers.txt"
    for i in range(5):
        storage.append_record(f"q{i}", "a", "b", path=target, max_bytes=0)
    assert list(tmp_path.glob("answers-*.txt")) == []
    text = target.read_text(encoding="utf-8")
    assert text.count("Record-ID:") == 5
