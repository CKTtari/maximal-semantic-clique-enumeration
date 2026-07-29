$ErrorActionPreference = 'Stop'

$utf8 = [System.Text.UTF8Encoding]::new($false)
$root = Get-Location
$bibPath = Join-Path $root 'references.bib'
$mainPath = Join-Path $root 'main.tex'

$bib = [System.IO.File]::ReadAllText($bibPath, $utf8)
$bib = [System.Text.RegularExpressions.Regex]::Replace(
    $bib,
    '(?ms)^@inproceedings\{pelleau2025hypergraph,.*?^\}\s*',
    '',
    1
)

$entries = @'

@article{yu2023fastqc,
  author  = {Kaiqiang Yu and Cheng Long},
  title   = {Fast Maximal Quasi-clique Enumeration: A Pruning and Branching Co-Design Approach},
  journal = {Proc. ACM Manag. Data},
  volume  = {1},
  number  = {3},
  pages   = {211:1--211:26},
  year    = {2023}
}

@inproceedings{wang2025hbbmc,
  author    = {Kaixin Wang and Kaiqiang Yu and Cheng Long},
  title     = {Maximal Clique Enumeration with Hybrid Branching and Early Termination},
  booktitle = {ICDE},
  pages     = {599--612},
  year      = {2025}
}

@article{wang2024kclique,
  author  = {Kaixin Wang and Kaiqiang Yu and Cheng Long},
  title   = {Efficient $k$-Clique Listing: An Edge-Oriented Branching Strategy},
  journal = {Proc. ACM Manag. Data},
  volume  = {2},
  number  = {1},
  pages   = {7:1--7:26},
  year    = {2024}
}

@article{gao2024kplex,
  author  = {Shuohao Gao and Kaiqiang Yu and Shengxin Liu and Cheng Long},
  title   = {Maximum $k$-Plex Search: An Alternated Reduction-and-Bound Method},
  journal = {Proc. VLDB Endow.},
  volume  = {18},
  number  = {2},
  pages   = {363--376},
  year    = {2024}
}

@article{dai2024biplex,
  author  = {Qiangqiang Dai and Rong-Hua Li and Donghang Cui and Meihao Liao and Yu-Xuan Qiu and Guoren Wang},
  title   = {Efficient Maximal Biplex Enumerations with Improved Worst-Case Time Guarantee},
  journal = {Proc. ACM Manag. Data},
  volume  = {2},
  number  = {3},
  pages   = {135:1--135:26},
  year    = {2024}
}

@article{fang2024iacs,
  author  = {Shuheng Fang and Kangfei Zhao and Yu Rong and Zhixun Li and Jeffrey Xu Yu},
  title   = {Inductive Attributed Community Search: To Learn Communities Across Graphs},
  journal = {Proc. VLDB Endow.},
  volume  = {17},
  number  = {10},
  pages   = {2576--2589},
  year    = {2024}
}

@article{wang2024transzero,
  author  = {Jianwei Wang and Kai Wang and Xuemin Lin and Wenjie Zhang and Ying Zhang},
  title   = {Efficient Unsupervised Community Search with Pre-trained Graph Transformer},
  journal = {Proc. VLDB Endow.},
  volume  = {17},
  number  = {9},
  pages   = {2227--2240},
  year    = {2024}
}

@inproceedings{yang2025similarweight,
  author    = {Jianye Yang and Lei Xing and Ziyi Ma and Xi Luo and Cuiyun Gao and Xuemin Lin},
  title     = {Maximal Similar-Weight Biclique Enumeration for Large Bipartite Graphs},
  booktitle = {ICDE},
  pages     = {1084--1097},
  year      = {2025}
}

@inproceedings{zhang2025fairclique,
  author    = {Qi Zhang and Rong-Hua Li and Zifan Zheng and Hongchao Qin and Ye Yuan and Guoren Wang},
  title     = {Efficient Maximum Fair Clique Search Over Large Networks},
  booktitle = {ICDE},
  pages     = {1043--1055},
  year      = {2025}
}

@inproceedings{li2025cliquecomparator,
  author    = {Xiaofan Li and Rui Zhou and Lu Chen and Chengfei Liu},
  title     = {Clique Comparator: A Fundamental Operator for Finding a Concise Clique Summary},
  booktitle = {ICDE},
  pages     = {3248--3260},
  year      = {2025}
}
'@

foreach ($key in @(
    'yu2023fastqc', 'wang2025hbbmc', 'wang2024kclique', 'gao2024kplex',
    'dai2024biplex', 'fang2024iacs', 'wang2024transzero',
    'yang2025similarweight', 'zhang2025fairclique', 'li2025cliquecomparator'
)) {
    if ($bib -match "(?m)^@\w+\{$key,") { throw "Duplicate bibliography key: $key" }
}
$bib = $bib.TrimEnd() + [Environment]::NewLine + $entries.Trim() + [Environment]::NewLine
[System.IO.File]::WriteAllText($bibPath, $bib, $utf8)

$main = [System.IO.File]::ReadAllText($mainPath, $utf8)
$related = @'
\section{Related Work}
\label{sec:related}

\stitle{Clique Enumeration and Listing.}
Moon and Moser established the combinatorial output bound for maximal cliques
\cite{moon1965}; BK backtracking~\cite{bron1973}, Tomita pivoting
\cite{tomita2006}, and degeneracy-oriented search~\cite{eppstein2010,eppstein2013}
form the standard practical foundation. Recent work improves branch selection
and termination for MCE~\cite{wang2025hbbmc}, develops edge-oriented
$k$-clique listing~\cite{wang2024kclique}, and summarizes large clique families
through comparison operators~\cite{li2025cliquecomparator}. These structural
techniques motivate the pivoted front end of \subEnum; MSC additionally couples
each search state to a non-monotone vector aggregate.

\stitle{Constrained Cohesive Subgraphs.}
Dense-subgraph models relax or augment the clique contract in different ways.
FastQC enumerates maximal quasi-cliques through co-designed pruning and
branching~\cite{yu2023fastqc}; recent exact methods address maximum $k$-plexes
\cite{gao2024kplex}, maximal biplexes~\cite{dai2024biplex}, fair cliques
\cite{zhang2025fairclique}, and similar-weight bicliques
\cite{yang2025similarweight}. MSC retains structural completeness and constrains
the group-average vector similarity, so these models provide complementary
output families. Our exact efficiency baseline instead preserves the same MSC
contract throughout search.

\stitle{Attributed Community Search.}
Attributed clustering and subgraph mining combine topology with attribute
homogeneity through clustering objectives, feature subspaces, or query-specific
cohesion~\cite{zhou2009,moser2009,gunnemann2010,silva2012,bothorel2015}.
Recent systems transfer attributed community search across graphs
\cite{fang2024iacs} and use pre-trained graph transformers for unsupervised
queries~\cite{wang2024transzero}. These methods return one or a small number of
query-oriented communities. MSC enumeration has a different data-management
contract: it returns every inclusion-wise maximal, structurally complete group
whose collective vector similarity reaches $\tau$.

\stitle{Vector Attributes.}
Vector-attributed graphs arise from knowledge-graph embeddings
\cite{nickel2011,bordes2013,dettmers2018,sun2019}, random-walk representations
\cite{perozzi2014deepwalk,grover2016}, and message-passing models
\cite{kipf2017,hamilton2017}. Approximate vector indexes accelerate
unconstrained nearest-neighbor retrieval~\cite{indyk1998,andoni2006,malkov2020,
johnson2021}. MSC evaluates similarities only on graph edges and preserves an
inclusive threshold exactly; our implementation therefore precomputes those
edge similarities and avoids approximation error inside enumeration.

'@
$start = $main.IndexOf('\section{Related Work}', [System.StringComparison]::Ordinal)
$end = $main.IndexOf('\section{Conclusion}', $start, [System.StringComparison]::Ordinal)
if ($start -lt 0 -or $end -lt 0) { throw 'Related Work markers not found' }
$main = $main.Substring(0, $start) + $related + $main.Substring($end)
[System.IO.File]::WriteAllText($mainPath, $main, $utf8)

Write-Output 'references.bib and Related Work updated successfully'
