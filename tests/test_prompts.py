from anli_classification.config import RunConfig, run_dir_for
from anli_classification.prompts import PROMPTS, NliPrompt, NliPromptV3
from anli_classification.schemas import Example

EXAMPLE = Example("dev", "r1", "u1", "A dog runs.", "An animal moves.", 0)


def test_v3_defines_labels_by_meaning_not_word_match():
    content = NliPromptV3().build(EXAMPLE)[1]["content"]
    assert "- entailment: The hypothesis follows from the premise, including through paraphrasing" in content
    assert "- neutral: The premise neither supports nor contradicts the hypothesis." in content
    assert "Do not choose neutral solely because the relationship is implicit." in content
    assert content.rstrip().endswith("entailment, neutral, or contradiction.")


def test_registry_maps_each_version_to_its_prompt():
    assert PROMPTS[NliPrompt.VERSION] is NliPrompt
    assert PROMPTS[NliPromptV3.VERSION] is NliPromptV3
    assert NliPrompt.VERSION != NliPromptV3.VERSION


def test_prompt_versions_write_to_separate_run_dirs(tmp_path):
    v2 = RunConfig(model="accounts/fireworks/models/m", output_dir=tmp_path, prompt_version=NliPrompt.VERSION)
    v3 = RunConfig(model="accounts/fireworks/models/m", output_dir=tmp_path)
    assert v2.run_dir == tmp_path / "m" / NliPrompt.VERSION
    assert v3.run_dir == tmp_path / "m" / NliPromptV3.VERSION
    assert run_dir_for(tmp_path, "accounts/fireworks/models/m", NliPrompt.VERSION) == v2.run_dir

