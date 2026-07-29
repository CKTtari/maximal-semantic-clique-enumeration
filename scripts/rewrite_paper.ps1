$ErrorActionPreference = 'Stop'

$paperPath = Join-Path (Get-Location) 'main.tex'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$text = [System.IO.File]::ReadAllText($paperPath, $utf8)

function Replace-Section {
    param(
        [Parameter(Mandatory = $true)][string]$Start,
        [Parameter(Mandatory = $true)][string]$End,
        [Parameter(Mandatory = $true)][string]$Replacement
    )
    $startIndex = $script:text.IndexOf($Start, [System.StringComparison]::Ordinal)
    if ($startIndex -lt 0) { throw "Start marker not found: $Start" }
    $endIndex = $script:text.IndexOf($End, $startIndex + $Start.Length, [System.StringComparison]::Ordinal)
    if ($endIndex -lt 0) { throw "End marker not found: $End" }
    $script:text = $script:text.Substring(0, $startIndex) + $Replacement + $script:text.Substring($endIndex)
}

function Replace-Exact {
    param(
        [Parameter(Mandatory = $true)][string]$Old,
        [Parameter(Mandatory = $true)][string]$New
    )
    if (-not $script:text.Contains($Old)) { throw "Exact text not found: $($Old.Substring(0, [Math]::Min(80, $Old.Length)))" }
    $script:text = $script:text.Replace($Old, $New)
}

$abstract = @'
\begin{abstract}
Maximal clique enumeration is a fundamental primitive for graph analysis, yet structural completeness alone cannot identify semantically coherent groups in vector-attributed graphs.  Adding a collective similarity constraint makes the problem substantially harder: average pairwise similarity is non-monotone over the subset lattice, so the pivot invariant behind scalable structural enumeration no longer holds.  We formalize maximal semantic clique (MSC) enumeration under inclusion-wise maximality and develop two exact algorithms.  \subEnum\ restores structural pivoting by separating structural MCE from bounded semantic subset enumeration.  \monoSemMCE\ targets selective thresholds through a canonical-parent theorem: deleting a minimum-contribution vertex from any feasible clique yields a unique feasible parent.  The resulting reverse search generates each feasible state once without visited tables or materialized size layers.  Across eight datasets and a fixed similarity-quantile grid, the two methods complete all 24 workloads, while the exact \semBK\ baseline reaches the 1,200-second limit on 15.  At $q_{99}$, \monoSemMCE\ is $2.92\times$--$6.24\times$ faster than \subEnum.  Multi-threshold output analysis further shows that MSCs provide a stronger semantic-purity--coverage tradeoff than pairwise-threshold clique mining.
\end{abstract}

'@
Replace-Section -Start '\begin{abstract}' -End '\maketitle' -Replacement $abstract

$contributions = @'
\stitleunderline{Problem formulation.}
We formalize maximal semantic clique enumeration under an inclusion-wise
maximal average-similarity constraint, establish its computational hardness,
and identify the non-monotonicity that prevents direct use of structural BK
pivoting.

\stitleunderline{Exact enumeration algorithms.}
We develop \subEnum, which separates structural MCE from bounded semantic
subset enumeration, and \monoSemMCE, whose canonical-parent relation organizes
high-threshold search without visited tables or materialized size layers. Both
algorithms preserve the complete MSC output set.

\stitleunderline{Large-scale evaluation.}
We evaluate exactness, runtime, memory, scalability, pruning mechanisms, and
output semantics on eight vector-attributed graphs under a fixed
similarity-quantile protocol. The results identify the operating regions of
the two search organizations and quantify their gains over the exact semantic
baseline.

'@
Replace-Section -Start '\stitle{Our Contributions.}' -End '\stitle{Paper Organization.}' -Replacement $contributions

Replace-Exact -Old '\mathbb{R}^d' -New '\mathbb{R}^D'
Replace-Exact -Old '$d$-dimensional attribute vector' -New '$D$-dimensional attribute vector'

$hardness = @'
\stitle{Hardness.}
MSC enumeration contains classical MCE as a special case.  When singleton
outputs are excluded and $\tau$ does not exceed the minimum similarity on any
edge, every structural maximal clique of size at least two satisfies the
semantic constraint, so the $\Omega(3^{n/3})$ output lower bound carries over.
The associated decision problem is NP-complete.

'@
Replace-Section -Start '\stitle{Hardness.}' -End '\begin{theorem}' -Replacement $hardness

$pivot = @'
\stitle{Why pivoting is not used.}
In standard BK, a pivot $p\in P\cup X$ guarantees that each structural maximal
clique in the current branch contains either $p$ or a vertex in
$P\setminus N(p)$.  Hence vertices in $P\cap N(p)$ need not become expansion
roots.  This argument depends on structural feasibility being preserved when
an adjacent pivot is added.

The average constraint invalidates that implication.  Consider a complete
graph on $\{s,p,u,w\}$ at the BK state $C=\{s\}$,
$P=\{p,u,w\}$, and $X=\emptyset$.  Let the three similarities within
$\{s,u,w\}$ equal $0.9$, the three similarities incident to $p$ equal $0$,
and $\tau=0.6$.  Then $\{s,u,w\}$ is an MSC with average $0.9$, whereas its
only strict clique superset has average $0.45$.  Choosing $p$ as the pivot
leaves only $p$ in $P\setminus N(p)$, so every explored child contains $p$
and the valid MSC $\{s,u,w\}$ is lost.  Pivot-free branching is therefore
required for the direct semantic BK baseline.

'@
Replace-Section -Start '\stitle{Why pivoting is not used.}' -End '\stitle{Complexity.}' -Replacement $pivot

$semComplexity = @'
\stitle{Complexity.}
\begin{theorem}[Time and Space Complexity]
  \label{the:alg1complexity}
  Let $d_G$ be the degeneracy of $G$, let $\mathcal{L}$ be the feasible
  terminal states reached by the outer search, and let $X_C$ be the candidate
  set examined by \textsc{Check} for $C\in\mathcal{L}$.  The running time is
  $O^*(n2^{d_G}+\sum_{C\in\mathcal{L}}2^{|X_C|})$, where $O^*$ suppresses
  polynomial set-intersection costs.  A graph-independent upper bound is
  $O^*(3^n)$.  With $p$ workers, the implementation uses
  $O(m+nD+pn^2)$ worst-case space.
\end{theorem}
\begin{proof}
  A degeneracy seed has at most $d_G$ later neighbors.  Without pivoting, its
  outer recursion may visit every subset of those neighbors, giving
  $O^*(n2^{d_G})$ states.  For a feasible terminal $C$, \textsc{Check} may
  independently visit every subset of $X_C$.  The same earlier extension can
  be reconsidered for several terminals; charging a state to a pair of nested
  vertex sets gives the safe $O^*(3^n)$ bound.  Similarities and sparse
  adjacency require $O(m+nD)$ storage.  Each worker owns an $O(n)$
  accumulator, while decreasing candidate arrays retained by a worst-case
  \textsc{Check} stack total $O(n^2)$.
\end{proof}

'@
Replace-Section -Start ('\stitle{Complexity.}' + [Environment]::NewLine + '\begin{theorem}[Time and Space Complexity]') -End '\stitle{Design Implication.}' -Replacement $semComplexity

$strsubComplexity = @'
\stitle{Characteristics and Applicable Regime.}
\subEnum\ applies Tomita pivoting during structural MCE, whose cost is
$O(d_Gn\,3^{d_G/3})$ and independent of $\tau$.  At low thresholds, large
feasible subsets are identified early within each structural host.  At higher
thresholds, the Top-Edge and vertex-contribution bounds skip infeasible size
layers.  Structural MCE remains a fixed front-end cost, which motivates the
canonical search developed next for selective thresholds.

\stitle{Complexity.}
\begin{theorem}[Time Complexity of \subEnum]
  \label{the:alg2complexity}
  For each structural maximal clique $M$, let $Q_M$ denote the number of exact
  local containment probes and let $b_M=\lceil |M|/w\rceil$ be its bitmap
  length in machine words.  \subEnum\ runs in
  \[
    \resizebox{0.99\columnwidth}{!}{$\displaystyle
      O\!\left(d_Gn\,3^{d_G/3}+\sum_{M\in\mathcal M}
      \left(2^{|M|}|M|^2+Q_Mb_M\right)
      +KN\log N+K\sum_{S\in\mathcal F}L_{\min}(S)\right)$}
  \]
  time, where $d_G$ is the degeneracy of $G$, $N=|\mathcal F|$ is the number
  of candidates before global deduplication,
  $K=\max_{S\in\mathcal F}|S|$, and
  $L_{\min}(S)=\min_{v\in S}|\mathit{inv}[v]|$.
\end{theorem}
\begin{proof}
  Tomita BK with degeneracy ordering costs
  $O(d_Gn\,3^{d_G/3})$~\cite{eppstein2013}.  For a host $M$ of size $h$,
  subset enumeration visits at most $2^h$ subsets and evaluates each from the
  cached similarity matrix in $O(h^2)$ time.  Exact local maximality performs
  $Q_M$ bitmap containment probes of $b_M$ words each; keeping this term
  explicit also covers instances with many feasible local subsets.  The final
  filter sorts $N$ candidates in $O(KN\log N)$ time.  It then scans the shortest
  inverted list for each $S$, with at most $L_{\min}(S)$ sorted-set containment
  checks of $O(K)$ time.
\end{proof}

\noindent\textbf{Remark.}
Without pruning, the enumeration term is exponential in each structural host,
and $Q_M$ is output sensitive.  The admissible bounds replace $2^{|M|}$ by
$\sum_{k=2}^{k^*}\binom{|M|}{k}$ when all larger layers are certified
infeasible; local containment reduces the candidates reaching the global
filter without changing the output set.

'@
Replace-Section -Start '\stitle{Characteristics and Applicable Regime.}' -End '\subsection{\monoSemMCE:' -Replacement ($strsubComplexity + '\subsection{\monoSemMCE:')

$terminalNoteOld = 'insertion paths before recursion rather than after state materialization.'
$terminalNoteNew = $terminalNoteOld + [Environment]::NewLine + 'The flag $\mathit{hasValid}$ is set before the ownership test because it records' + [Environment]::NewLine + 'the existence of any feasible one-vertex extension, not the existence of an' + [Environment]::NewLine + 'owned child. A state with such an extension cannot itself be inclusion-wise' + [Environment]::NewLine + 'maximal; the extension is generated from its unique canonical parent.'
Replace-Exact -Old $terminalNoteOld -New $terminalNoteNew
Replace-Exact -Old ('\begin{theorem}[Enumeration Correctness]' + [Environment]::NewLine + 'Algorithm~\ref{alg:monosem-mce}') -New ('\begin{theorem}[Enumeration Correctness]' + [Environment]::NewLine + '\label{thm:monosem-correctness}' + [Environment]::NewLine + 'Algorithm~\ref{alg:monosem-mce}')

$monoComplexity = @'
\stitle{Complexity.}
\begin{theorem}[Time and Space Complexity of \monoSemMCE]
  \label{thm:monosem-complexity}
  Let $F_\tau$ be the number of feasible cliques of size at least two, $K$ the
  maximum clique size, $\Delta$ the maximum degree, $T$ the number of workers,
  and $N=|\mathcal F|$ the number of terminal candidates.  A conservative
  bound for canonical search is
  \[
    O\!\left(F_\tau\Delta(\Delta+K)\log\Delta
      +KN\log N+K\sum_{S\in\mathcal F}L_{\min}(S)\right),
  \]
  with $O(m+T(n+K\Delta)+NK)$ words of space in addition to vector storage.
\end{theorem}
\begin{proof}
  Canonical ownership bounds accepted states by $F_\tau$.  A state tests at
  most $\Delta$ candidates.  For each candidate, common-neighbor intersection
  and accumulator maintenance cost $O(\Delta)$ conservatively, while updating
  and comparing at most $K$ member contributions uses cached similarities with
  $O(\log\Delta)$ lookup.  This is bounded by
  $O(\Delta(\Delta+K)\log\Delta)$ per accepted state.  The final filter has the
  same $O(KN\log N+K\sum_S L_{\min}(S))$ cost as in \subEnum.

  Sparse similarity adjacency and neighbor bitmaps use $O(m)$ words.  Each
  worker owns an $O(n)$ generation-tagged accumulator and $O(K\Delta)$ DFS
  scratch space.  Terminal candidates occupy $O(NK)$ words; no feasible-size
  layer or visited-state table is materialized.
\end{proof}

\noindent\textbf{Remark.}
Increasing $\tau$ contracts the feasible-state family $F_\tau$, which gives
\monoSemMCE\ its selective-threshold operating region.  \subEnum\ instead pays
the structural MCE cost and is less sensitive to $F_\tau$.

'@
Replace-Section -Start ('\stitle{Complexity.}' + [Environment]::NewLine + '\begin{theorem}[Time and Space Complexity of \monoSemMCE]') -End '\section{Implementation Optimizations}' -Replacement ($monoComplexity + '\section{Implementation Optimizations}')

$experiments = @'
\section{Experiments}
\label{sec:exp}

We organize the evaluation around five research questions. RQ1 evaluates exact
end-to-end performance; RQ2 identifies threshold and scale regimes; RQ3
isolates parallel and pruning mechanisms; RQ4 measures how the MSC output space
changes with $\tau$; and RQ5 examines semantic quality against structurally
defined alternatives.

\subsection{Experimental Setup}
\label{sec:exp-setup}

\stitle{Methods and comparison scope.}
\StructBK\ is a Tomita-pivot structural MCE implementation and supplies only a
threshold-independent structural-time reference. \semBK\ is the exact direct
baseline: it uses the same MSC definition, threshold, similarity values, and
strict-containment filter as \subEnum\ and \monoSemMCE. Consequently, runtime
speedups are claimed only among these three exact MSC methods. \pairSim\ keeps
edges whose endpoint similarity reaches $\tau$ and then runs structural MCE;
FastQC enumerates maximal quasi-cliques under its degree-density constraint
\cite{yu2023fastqc}. These two methods return different pattern families and
are used only for quality--coverage comparisons.

\stitle{Datasets.}
The common grid contains three graphs with natural attributes and five public
graphs with reproducibly generated attributes. FB15K-237 descriptions use
all-MiniLM-L6-v2 embeddings; GitHub-Social uses its official 4,005-dimensional
developer features; and ogbn-arxiv retains its official 128-dimensional node
features~\cite{toutanova2015observed,wang2020ogb}. For email-Enron and the four
SuiteSparse graphs~\cite{leskovec2014snap,davis2011uf}, PCG64 draws 256
coordinates independently from $U[-1,1)$ under fixed dataset seeds, followed
by $\ell_2$ normalization. Graph structure is never synthesized.

\begin{table}[t]
\scriptsize
\centering
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{@{}lrrrl@{}}
\toprule
Dataset & $|V|$ & $|E|$ & $D$ & Attributes \\
\midrule
FB15K-237 & 14,541 & 248,611 & 384 & natural \\
GitHub-Social & 37,700 & 289,003 & 4,005 & natural \\
ogbn-arxiv & 169,343 & 1,157,799 & 128 & natural \\
email-Enron & 36,692 & 183,831 & 256 & generated \\
sc-nasasrb & 54,870 & 1,311,227 & 256 & generated \\
sc-pkustk11 & 87,804 & 2,565,054 & 256 & generated \\
sc-pwtk & 217,891 & 5,653,221 & 256 & generated \\
sc-ldoor & 909,537 & 20,770,807 & 256 & generated \\
\bottomrule
\end{tabular}
\caption{[RQ1--RQ5] Eight public graphs cover natural and controlled vector attributes.}
\label{tab:datasets}
\end{table}

\stitle{Protocol.}
Experiments run on an AMD Ryzen~9 7945HX workstation with 16 physical cores,
32 hardware threads, and 32\,GiB memory under Windows~11 and MSYS2 GCC~13.2.
Each process is limited to 16\,GiB and 32 OpenMP threads. The uniform timeout is
1,200 seconds. Completed runtime configurations are repeated three times and
reported by the median; a timeout is recorded after one complete limit window.
Runtime binaries contain no statistics counters.

Absolute cosine values are not comparable across vector sources. We therefore
derive thresholds from the edge-similarity distribution of each dataset. The
common grid fixes $q_{20}$, $q_{80}$, and $q_{99}$ before execution; sensitivity
experiments use $q_{5}$, $q_{20}$, $q_{50}$, $q_{70}$, $q_{80}$, $q_{90}$,
$q_{95}$, $q_{97}$, and $q_{99}$. All exact methods receive the same stored
float32 edge similarities and evaluate inclusive feasibility using
double-precision sums.

\subsection{RQ1: Exactness and End-to-End Efficiency}
This experiment answers RQ1 by comparing complete MSC output sets and total
runtime under the common grid. Exhaustive oracles first test 225,000
vertex-bound instances and 40,001 canonical-parent instances, including
negative similarities, equality at $\tau$, and tied minimum contributions.
On real graphs, jointly completed methods agree on normalized clique sets,
counts, and size histograms. In particular, all three exact methods return
263,721, 71,118, and 2,636 MSCs on GitHub-Social at $q_{20}$, $q_{80}$, and
$q_{99}$, respectively; \subEnum\ and \monoSemMCE\ also agree on every jointly
completed row of Table~\ref{tab:main-runtime}.

\begin{table*}[t]
\scriptsize
\centering
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.02}
\begin{tabular*}{0.98\textwidth}{@{\extracolsep{\fill}}lcrrrrr@{}}
\toprule
Dataset & $q$ & MSCs & \shortstack{\StructBK\\structure} & \semBK & \subEnum & \monoSemMCE \\
\midrule
FB15K-237 & 20 & 206,470 & \multirow{3}{*}{\cellcolor{gray!12}0.103} & TLE & \cellcolor{bestgreen}\textbf{0.221} & TLE \\
 & 80 & 76,106 & & TLE & \cellcolor{bestgreen}\textbf{9.132} & TLE \\
 & 99 & 7,476 & & TLE & 45.234 & \cellcolor{bestgreen}\textbf{10.091} \\
\midrule
GitHub-Social & 20 & 263,721 & \multirow{3}{*}{\cellcolor{gray!12}0.110} & 55.269 & \cellcolor{bestgreen}\textbf{0.327} & 29.315 \\
 & 80 & 71,118 & & 44.161 & \cellcolor{bestgreen}\textbf{0.314} & 20.551 \\
 & 99 & 2,636 & & 38.213 & 0.391 & \cellcolor{bestgreen}\textbf{0.118} \\
\midrule
ogbn-arxiv & 20 & 925,239 & \multirow{3}{*}{\cellcolor{gray!12}0.309} & 2.783 & \cellcolor{bestgreen}\textbf{1.233} & 3.357 \\
 & 80 & 226,936 & & 1.206 & \cellcolor{bestgreen}\textbf{0.872} & 1.610 \\
 & 99 & 10,261 & & 1.318 & 0.843 & \cellcolor{bestgreen}\textbf{0.197} \\
\midrule
email-Enron & 20 & 223,703 & \multirow{3}{*}{\cellcolor{gray!12}0.090} & 18.330 & \cellcolor{bestgreen}\textbf{0.259} & 16.739 \\
 & 80 & 60,352 & & 21.369 & 1.266 & \cellcolor{bestgreen}\textbf{0.083} \\
 & 99 & 1,819 & & 9.608 & 0.206 & \cellcolor{bestgreen}\textbf{0.033} \\
\midrule
sc-nasasrb & 20 & 28,154 & \multirow{3}{*}{\cellcolor{gray!12}0.090} & TLE & \cellcolor{bestgreen}\textbf{0.301} & TLE \\
 & 80 & 834,914 & & TLE & 102.820 & \cellcolor{bestgreen}\textbf{0.759} \\
 & 99 & 12,742 & & TLE & 0.415 & \cellcolor{bestgreen}\textbf{0.079} \\
\midrule
sc-pkustk11 & 20 & 22,612 & \multirow{3}{*}{\cellcolor{gray!12}0.143} & TLE & \cellcolor{bestgreen}\textbf{0.645} & TLE \\
 & 80 & 2,214,411 & & TLE & TLE & \cellcolor{bestgreen}\textbf{2.219} \\
 & 99 & 24,891 & & TLE & 0.611 & \cellcolor{bestgreen}\textbf{0.167} \\
\midrule
sc-pwtk & 20 & 58,523 & \multirow{3}{*}{\cellcolor{gray!12}0.286} & TLE & \cellcolor{bestgreen}\textbf{1.093} & TLE \\
 & 80 & 4,483,887 & & TLE & TLE & \cellcolor{bestgreen}\textbf{3.977} \\
 & 99 & 54,907 & & TLE & 1.364 & \cellcolor{bestgreen}\textbf{0.421} \\
\midrule
sc-ldoor & 20 & 274,585 & \multirow{3}{*}{\cellcolor{gray!12}1.679} & TLE & \cellcolor{bestgreen}\textbf{4.870} & TLE \\
 & 80 & 13,473,317 & & TLE & 563.996 & \cellcolor{bestgreen}\textbf{14.944} \\
 & 99 & 202,030 & & TLE & 4.845 & \cellcolor{bestgreen}\textbf{1.658} \\
\bottomrule
\end{tabular*}
\caption{[RQ1] Median end-to-end seconds; the two specialized methods jointly complete all 24 MSC workloads.}
\label{tab:main-runtime}
\end{table*}

Table~\ref{tab:main-runtime} shows that \subEnum\ completes 22 of 24 workloads
and every $q_{20}$ row, while \monoSemMCE\ resolves the two intermediate rows
where host-clique subset enumeration reaches TLE. At $q_{99}$, canonical search
is faster on all eight graphs, with speedups from $2.92\times$ on sc-ldoor to
$6.24\times$ on email-Enron. \semBK\ completes all nine natural-attribute rows
except FB15K-237, but its pivot-free search is slower than the best specialized
method on every completed row. The gray \StructBK\ column is reported only to
separate structural enumeration cost from semantic processing.

\subsection{RQ2: Threshold and Scale Regimes}
This experiment answers RQ2 by varying selectivity independently of graph size.
Figure~\ref{fig:threshold-runtime} uses the full fixed quantile grid. On
ogbn-arxiv, both specialized methods complete the sweep and cross near
$q_{95}$; on sc-ldoor, \subEnum\ covers low and intermediate thresholds, while
\monoSemMCE\ becomes dominant once the feasible canonical tree contracts.
TLE points are shown at the fixed limit and are not interpolated.

\begin{figure}[t]
\centering
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/threshold_runtime_arxiv.pdf}
\caption{ogbn-arxiv}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/threshold_runtime_ldoor.pdf}
\caption{sc-ldoor}
\end{subfigure}
\caption{[RQ2] Fixed quantile sweeps expose broad \subEnum\ coverage and selective \monoSemMCE\ speedups.}
\Description{Runtime over nine fixed threshold quantiles on ogbn-arxiv and sc-ldoor.}
\label{fig:threshold-runtime}
\end{figure}

Figure~\ref{fig:scale-regime} repeats the comparison on nested induced
ogbn-arxiv graphs whose vertices follow one fixed degree order. The crossover
remains at $q_{95}$ from 25K vertices through the full graph. At $q_{99}$ on
the full graph, \subEnum\ and \monoSemMCE\ take 0.707 and 0.200 seconds and
return the same 10,261 MSCs. The stable boundary links the regime change to
semantic selectivity rather than one graph size.

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{tex-data/fig/scale_regime.pdf}
\caption{[RQ2] The \monoSemMCE\ high-threshold region persists across four nested graph sizes.}
\Description{Runtime ratio between StrSub and MonoSemMCE over graph sizes and threshold quantiles.}
\label{fig:scale-regime}
\end{figure}

\subsection{RQ3: Parallelism and Search Mechanisms}
This experiment answers RQ3 by connecting parallel speedup to measured search
work. Figure~\ref{fig:parallel-scaling} reports one representative workload for
each specialized algorithm. Dynamic edge-root scheduling gives \monoSemMCE\
$5.62\times$ speedup at 32 threads. Fixed-size subset partitioning gives
\subEnum\ $2.39\times$ at 16 threads and $2.26\times$ at 32 threads. Separately
collected phase counters attribute 98.88\% of its 32-thread work to host-clique
subset processing, while the final global filter takes 65 ms; the remaining
saturation is therefore caused by unequal host-clique tasks.

\begin{figure}[t]
\centering
\includegraphics[width=0.84\linewidth]{tex-data/fig/parallel_scaling.pdf}
\caption{[RQ3] Independent root scheduling yields parallel gains for both specialized searches.}
\Description{Speedup from one to 32 threads for representative StrSub and MonoSemMCE workloads.}
\label{fig:parallel-scaling}
\end{figure}

Figure~\ref{fig:mechanisms} relates runtime regimes to work counters collected
outside timing runs. On sc-nasasrb at $q_{99}$, Top-Edge pruning reduces subset
tests from 16.50 to 12.95 million, and the vertex bound reduces them from 80.28
to 12.95 million. On ogbn-arxiv, accepted canonical states contract from 17.08
million at $q_5$ to 18,011 at $q_{99}$.

\begin{figure}[t]
\centering
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/mechanism_strsub.pdf}
\caption{\subEnum\ pruning}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/mechanism_monosem.pdf}
\caption{\monoSemMCE\ states}
\end{subfigure}
\caption{[RQ3] Pruning bounds and canonical-tree contraction remove the dominant search work.}
\Description{Subset tests by StrSub variant and accepted MonoSemMCE states by threshold.}
\label{fig:mechanisms}
\end{figure}

\begin{table}[t]
\scriptsize
\centering
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llrr@{}}
\toprule
Component & Metric / workload & Full & Removed \\
\midrule
Top-Edge & tests (M) / nasa $q_{99}$ & \cellcolor{bestgreen}\textbf{12.95} & 16.50 \\
Vertex bound & tests (M) / nasa $q_{99}$ & \cellcolor{bestgreen}\textbf{12.95} & 80.28 \\
Local containment & GiB / FB15K $q_{80}$ & \cellcolor{bestgreen}\textbf{0.12} & 2.26 \\
\bottomrule
\end{tabular}
\caption{[RQ3] Exact \subEnum\ ablations reduce subset work or candidate memory without changing outputs.}
\label{tab:ablation}
\end{table}

The corresponding runtimes are 0.287 versus 0.324 seconds for Top-Edge,
0.287 versus 0.312 seconds for the vertex bound, and 8.060 versus 11.685
seconds for local containment. Table~\ref{tab:generation-ablation} then holds
the feasibility test, similarity cache, and final filter fixed while replacing
only duplicate-state organization. Canonical ownership is fastest and uses the
least memory on both workloads; every variant returns the same normalized MSC
set and size histogram.

\begin{table}[t]
\scriptsize
\centering
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{ogbn $q_{95}$} & \multicolumn{2}{c}{nasa $q_{80}$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Generation & s & GiB & s & GiB \\
\midrule
\rowcolor{bestgreen}Canonical & \textbf{0.388} & \textbf{0.39} & \textbf{0.729} & \textbf{0.22} \\
Visited DFS & 2.432 & 0.53 & 3.069 & 0.64 \\
Layered & 1.635 & 0.49 & 3.922 & 0.81 \\
\bottomrule
\end{tabular}
\caption{[RQ3] Canonical ownership improves both runtime and memory over visited and layered generation.}
\label{tab:generation-ablation}
\end{table}

\subsection{RQ4: MSC Output Space}
This experiment answers RQ4 by measuring exact output counts and mean sizes on
FB15K-237, ogbn-arxiv, sc-nasasrb, and sc-ldoor over
$q_{20},q_{50},q_{80},q_{95},q_{99}$. Figure~\ref{fig:result-distribution}
normalizes each value by \StructBK\ on the same graph. The geometric-mean count
ratios are 1.04, 7.50, 3.53, 0.61, and 0.11: intermediate thresholds split
large structural cliques into multiple inclusion-maximal feasible subsets,
whereas selective thresholds contract the result family. Mean size decreases
from 1.06 to 0.39 of the structural reference over the same grid.

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{tex-data/fig/result_distribution_legend.pdf}
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/result_distribution_count.pdf}
\caption{Output count}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/result_distribution_size.pdf}
\caption{Mean size}
\end{subfigure}
\caption{[RQ4] MSCs expand at intermediate thresholds and contract sharply in the selective regime.}
\Description{Normalized MSC count and mean size over five threshold quantiles on four datasets.}
\label{fig:result-distribution}
\end{figure}

\subsection{RQ5: Semantic Quality and External Pattern Families}
This experiment answers RQ5 with independent labels rather than the similarity
objective itself. The full ogbn-arxiv sweep compares MSC with \pairSim\ under
the same $q_{20},q_{50},q_{80},q_{95},q_{99}$ thresholds; \StructBK\ is a
single threshold-independent reference. Purity is the largest official-label
fraction in an output, and coverage is the fraction of vertices occurring in
at least one non-singleton output. Figure~\ref{fig:quality-threshold} shows the
complete trajectory. From $q_{50}$ through $q_{99}$, MSC has both higher mean
purity and higher coverage than \pairSim; at $q_{20}$ their purity differs by
0.002 while MSC covers 0.5 percentage points more vertices. Thus the conclusion
is based on a fixed threshold range rather than a selected operating point.

\begin{figure}[t]
\centering
\includegraphics[width=0.90\linewidth]{tex-data/fig/quality_threshold_sweep.pdf}
\caption{[RQ5] Across the fixed threshold grid, MSC improves the purity--coverage frontier over \pairSim.}
\Description{Purity and vertex coverage of MSC, PairSim-MCE, and StructBK on ogbn-arxiv.}
\label{fig:quality-threshold}
\end{figure}

We next compare four pattern families at the pre-registered $q_{80}$ common-grid
point on GitHub-Social and an induced 25K-vertex ogbn-arxiv graph. To make size
support comparable, all metrics retain outputs of size at least ten. FastQC
uses $\gamma=0.9$ and minimum size ten, a fixed dense quasi-clique operating
point rather than a parameter claimed equivalent to $\tau$. Accordingly,
Figure~\ref{fig:multi-baseline-quality} compares output semantics, not runtime.
On GitHub-Social, MSC reaches 0.964 purity and covers $6.18\times$ as many
vertices as \pairSim. On ogbn-arxiv-25K, MSC reaches 0.992 purity, compared with
0.973 for \StructBK\ and 0.342 for FastQC, while covering $5.39\times$ as many
vertices as \pairSim. The two datasets are shown separately because their label
spaces and coverage scales are not commensurate.

\begin{figure}[t]
\centering
\includegraphics[width=0.90\linewidth]{tex-data/fig/multi_baseline_quality_legend.pdf}
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/multi_baseline_quality_github.pdf}
\caption{GitHub-Social}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
\centering
\includegraphics[width=\linewidth]{tex-data/fig/multi_baseline_quality_ogbn25k.pdf}
\caption{ogbn-arxiv-25K}
\end{subfigure}
\caption{[RQ5] MSC balances label purity and coverage against three structurally defined pattern families.}
\Description{Purity and coverage for MSC, PairSim-MCE, FastQC, and StructBK on two labeled datasets.}
\label{fig:multi-baseline-quality}
\end{figure}

Figure~\ref{fig:quality-case} provides a deterministic instance-level view: it
selects the largest label-pure MSC at ogbn-arxiv $q_{80}$. All 18 papers belong
to \texttt{cs.IT} and concern private information retrieval. Their average
similarity is 0.919 at $\tau=0.889$, although 20 of 153 pairs fall below
$\tau$. The average constraint therefore retains a collectively coherent
complete subgraph that the every-edge threshold separates.

\begin{figure}[t]
\centering
\includegraphics[width=0.90\linewidth]{tex-data/fig/quality_case_network.pdf}
\caption{[RQ5] An 18-paper MSC preserves a coherent research group despite 20 sub-threshold pairs.}
\Description{A paper network showing semantic-neighbor links and recurrent authors in one MSC.}
\label{fig:quality-case}
\end{figure}

\FloatBarrier
'@
Replace-Section -Start '\section{Experiments}' -End '\section{Related Work}' -Replacement ($experiments + "`r`n\section{Related Work}")

[System.IO.File]::WriteAllText($paperPath, $text, $utf8)
Write-Output 'main.tex rewritten successfully'
