"""Editing a live labeling config without disturbing its layout.

The config may have been hand-tuned in the Label Studio UI, so the tests
that matter here are about what an edit leaves alone.
"""

import pytest

from auto_labeller.label_config import (
    LabelConfigError,
    add_class,
    find_control,
    get_classes,
)

CHOICES = """<View style="display: flex;">
  <View style="flex: 1;">
    <Image name="image" value="$image" zoom="true"/>
  </View>
  <Choices name="label" toName="image" choice="multiple">
    <Choice value="cat" hotkey="1"/>
    <Choice value="dog" hotkey="2"/>
  </Choices>
</View>
"""


def test_find_control_and_get_classes():
    assert find_control(CHOICES) == ("Choices", "Choice")
    assert get_classes(CHOICES) == ["cat", "dog"]


def test_find_control_reports_what_it_looked_for():
    with pytest.raises(LabelConfigError, match="No class control found"):
        find_control("<View><Text name='t' value='$text'/></View>")


def test_add_class_changes_only_the_inserted_line():
    updated = add_class(CHOICES, "bird")
    added = set(updated.splitlines()) - set(CHOICES.splitlines())
    assert added == {'    <Choice value="bird" hotkey="3"/>'}
    # Every original line survives, in order: the hand-tuned layout is intact
    assert [x for x in updated.splitlines() if x in CHOICES.splitlines()] == CHOICES.splitlines()


def test_add_class_inserts_before_the_closing_control_tag():
    updated = add_class(CHOICES, "bird").splitlines()
    assert updated.index('    <Choice value="bird" hotkey="3"/>') < updated.index("  </Choices>")


def test_add_class_copies_the_indentation_of_the_last_class():
    deep = CHOICES.replace('    <Choice value="dog"', '        <Choice value="dog"')
    assert '        <Choice value="bird"' in add_class(deep, "bird")


def test_hotkeys_continue_from_the_highest_in_use():
    assert 'hotkey="3"' in add_class(CHOICES, "bird")


def test_no_hotkey_is_offered_when_the_config_does_not_use_them():
    # A config that deliberately avoids hotkeys keeps avoiding them
    without = CHOICES.replace(' hotkey="1"', "").replace(' hotkey="2"', "")
    assert "hotkey" not in add_class(without, "bird")


def test_no_hotkey_when_only_some_classes_have_one():
    partial = CHOICES.replace(' hotkey="2"', "")
    assert 'value="bird"/>' in add_class(partial, "bird")


def test_hotkeys_stop_after_the_ninth_class():
    nine = "\n".join(f'    <Choice value="c{i}" hotkey="{i}"/>' for i in range(1, 10))
    xml = f'<View><Choices name="label" toName="image">\n{nine}\n  </Choices></View>'
    # Label Studio only binds single digits
    assert 'value="c10"/>' in add_class(xml, "c10")


def test_a_duplicate_class_is_refused():
    with pytest.raises(LabelConfigError, match="already in the labeling config"):
        add_class(CHOICES, "cat")


@pytest.mark.parametrize("name", ["a<b", "a>b", 'a"b', "a&b"])
def test_names_that_would_break_the_xml_are_refused(name):
    with pytest.raises(LabelConfigError, match="not valid in XML"):
        add_class(CHOICES, name)


def test_works_on_a_bbox_config_too():
    xml = (
        '<View><Image name="image" value="$image"/>\n'
        '<RectangleLabels name="label" toName="image">\n'
        '    <Label value="cat" hotkey="1"/>\n'
        "  </RectangleLabels></View>"
    )
    assert find_control(xml) == ("RectangleLabels", "Label")
    assert '<Label value="dog" hotkey="2"/>' in add_class(xml, "dog")
