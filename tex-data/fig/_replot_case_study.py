#!/usr/bin/env python3
"""只重画 case_study_pir，跳过其他图表。"""
from plot_paper_figures import (
    configure_style,
    case_study_pir,
)

if __name__ == "__main__":
    configure_style()
    case_study_pir()
    print("Re-generated case_study_pir.pdf")
