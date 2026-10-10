"""Apply the authorized narrative cleanup; preserve a pre-edit source backup."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'Thesis/acl_latex.tex'
backup=ROOT/'.local_backups/paper_before_reviewer_revision_20261010'
backup.mkdir(parents=True,exist_ok=True)
if (backup/p.name).exists(): raise RuntimeError('Already backed up; review rather than reapply')
shutil.copy2(p,backup/p.name)
pdf=p.with_suffix('.pdf')
if pdf.exists(): shutil.copy2(pdf,backup/pdf.name)
s=p.read_text(encoding='utf-8')
s=s.replace("Early cooperation need not persist: Qwen reaches zero late cooperation without late fallbacks, while earlier fallback-affected histories qualify interpretation of its trajectory. A two-example audit additionally illustrates action--explanation inconsistency in valid first-attempt outputs, with unknown prevalence.","Early cooperation need not persist: DeepSeek's free-form cooperation falls from 39.87\\% in rounds 1--5 to 2.79\\% in rounds 26--30.")
s=s.replace('persistence, and action--explanation consistency','and persistence')
s=s.replace('Execution reliability and a two-example explanation audit further qualify what can be inferred from agent outputs.','Execution reliability qualifies what can be inferred from the observed trajectories.')
s=s.replace('(iii) a reliability and explanation case study identifying pipeline and rationale pitfalls through auditable examples, without estimating their prevalence.','(iii) an execution-reliability analysis identifying where pipeline failures limit behavioral interpretation.')
start=s.index(' Returned explanations supply an additional object of inspection:')
end=s.index('\n',start)
s=s[:start]+s[end:]
s=s.replace('\\subsection{Reliability and Action--Explanation Consistency}','\\subsection{Execution Reliability}')
s=s.replace('\\textbf{Execution audits and action checks are necessary to interpret\ncooperation.}','\\textbf{Execution reliability is necessary to interpret cooperation.}')
start=s.index('Selected valid first-attempt outputs also contradict')
end=s.index('Exploratory E1',start)
s=s[:start]+s[end:]
s=s.replace('The level, response, and persistence results answer different questions; the two-example audit separately checks whether explanations agree with actions.','The level, response, and persistence results answer different questions.')
s=s.replace(' A valid parse also does not ensure that an explanation describes the selected action.','')
s=s.replace(' Audit rationales against actions, because valid first-attempt outputs can be inconsistent; an example audit must not be reported as a prevalence estimate.','')
s=s.replace('reasoning. H5 was not evaluated; the export and planned human-coding protocol are preserved in Appendix~\\ref{sec:coding}. Temporal and qualitative','reasoning. Temporal')
s=s.replace('Future explanation coding should document independent rater instructions\nand agreement before reporting mechanism claims.\n','')
start=s.index('\\section{Explanation Coding Protocol (Not Evaluated)}')
end=s.index('\\section{Qwen Saved-Log Checks',start)
s=s[:start]+s[end:]
start=s.index('The action--explanation examples in Section~')
end=s.index('\n\n',start)
s=s[:start]+s[end+2:]
# Place the existing measurement grid alongside the contribution statement.
s=s.replace('\\input{tables/measurement_grid}\n','')
anchor='The evaluation targets measurement of agent behavior, rather than a universal scaling law or a communication mechanism.'
assert anchor in s
s=s.replace(anchor,anchor+'\n\n\\input{tables/measurement_grid}',1)
p.write_text(s,encoding='utf-8')
print('Revised manuscript; original source and PDF backed up:',backup)
