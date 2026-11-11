#pragma once

#include <algorithm>

namespace simple_ai_trading::presentation {

// Client-area DIPs. Painting and child windows consume the same section bounds.
struct WorkspaceLayout {
    int main_left = 254;
    int right;
    int width;
    int header_height = 96;
    int footer_top;
    int bottom;
    int command_top = 176;
    int command_height = 164;
    int workflow_label_top;
    int workflow_top;
    int workflow_columns;
    int workflow_rows;
    int workflow_gap = 12;
    int workflow_height = 52;
    int safety_label_top;
    int safety_top;
    int safety_gap = 10;
    int safety_height = 40;
    int diagnostic_top;
    int output_card_top;
    int output_label_top;
    int output_top;
};

inline WorkspaceLayout workspace_layout(int client_width, int client_height, int action_count) {
    WorkspaceLayout result{};
    result.right = client_width - 20;
    result.width = result.right - result.main_left;
    result.footer_top = client_height - 56;
    result.bottom = result.footer_top - 20;
    result.workflow_label_top = result.command_top + result.command_height + 14;
    result.workflow_top = result.workflow_label_top + 28;
    result.workflow_columns = result.width >= 1040 ? 4 : 2;
    const int visible = std::clamp(action_count, 0, 4);
    result.workflow_rows = (visible + result.workflow_columns - 1) / result.workflow_columns;
    const int workflow_extent = result.workflow_rows == 0 ? 0
        : result.workflow_rows * result.workflow_height + (result.workflow_rows - 1) * result.workflow_gap;
    result.safety_label_top = result.workflow_top + workflow_extent + 14;
    result.safety_top = result.safety_label_top + 28;
    result.diagnostic_top = result.safety_top + result.safety_height + result.safety_gap;
    result.output_card_top = result.diagnostic_top + result.safety_height + 14;
    result.output_label_top = result.output_card_top + 10;
    result.output_top = result.output_label_top + 24;
    return result;
}

}  // namespace simple_ai_trading::presentation
