"""Golden evaluation set over the sample corpus.

Relevance is defined *objectively* rather than by hand-labelling chunk ids: a
retrieved chunk is relevant if it literally contains ``gold_phrase``, a short
distinctive string that occurs verbatim in the source PDF.  This means the
labels survive any change to chunk size, overlap or the parser, so the harness
keeps measuring retrieval quality instead of measuring chunker churn.

``answer_must_contain`` is checked against the generated answer, which is a
stricter, end-to-end signal than retrieval alone.

The ``OUT_OF_CORPUS`` questions have no answer in the documents. They exist to
measure the refusal gate: a RAG system that never says "I don't know" is not
trustworthy, and refusal behaviour has to be quantified like anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalCase:
    """One graded question."""

    question: str
    gold_phrase: str = ""
    answer_must_contain: list[str] = field(default_factory=list)
    doc_hint: str = ""
    category: str = "in_corpus"

    @property
    def is_out_of_corpus(self) -> bool:
        return self.category == "out_of_corpus"


CASE_LAW: list[EvalCase] = [
    EvalCase(
        "What was the total consideration for the flat?",
        "total consideration of Rs. 1,42,50,000",
        ["1,42,50,000"],
        "case",
    ),
    EvalCase("Which flat number is in dispute?", "Flat No. B-1104", ["B-1104"], "case"),
    EvalCase("When was the judgment pronounced?", "14 March 2023", ["14 March 2023"], "case"),
    EvalCase(
        "What rate of interest was awarded on the refund?",
        "9.5 per cent per annum",
        ["9.5"],
        "case",
    ),
    EvalCase(
        "How much compensation was awarded for mental agony?",
        "compensation of Rs. 5,00,000 for mental agony",
        ["5,00,000"],
        "case",
    ),
    EvalCase("What costs were awarded in the appeal?", "quantified at Rs. 75,000", ["75,000"], "case"),
    EvalCase(
        "Within how many days must the respondent pay?",
        "within ninety days",
        ["ninety days"],
        "case",
    ),
    EvalCase(
        "What is the default rate of interest if payment is delayed?",
        "12 per cent per annum from the date of default",
        ["12 per cent"],
        "case",
    ),
    EvalCase("How much had the appellant already paid?", "Rs. 1,28,25,000", ["1,28,25,000"], "case"),
    EvalCase(
        "What was the contractual date for handing over possession?",
        "on or before 31 December 2019",
        ["31 December 2019"],
        "case",
    ),
    EvalCase(
        "When was the occupation certificate granted?",
        "granted by the Municipal Corporation of Greater Mumbai only on 19 November 2022",
        ["19 November 2022"],
        "case",
    ),
    EvalCase(
        "Which precedent was relied on for the unconditional right to refund?",
        "Newtech Promoters and",
        ["Newtech"],
        "case",
    ),
    EvalCase(
        "Which judgment was followed on the question of who is a consumer?",
        "Imperia Structures",
        ["Imperia"],
        "case",
    ),
    EvalCase(
        "How many months of delay did the court find unexplained?",
        "unexplained delay of twenty-two months",
        ["twenty-two months"],
        "case",
    ),
    EvalCase(
        "What monthly rent was the appellant paying for rented premises?",
        "monthly rent of Rs. 62,000",
        ["62,000"],
        "case",
    ),
]

LEASE_DEED: list[EvalCase] = [
    EvalCase(
        "What is the monthly rent for the demised premises?",
        "monthly rent for the demised premises shall be Rs. 48,500",
        ["48,500"],
        "lease",
    ),
    EvalCase(
        "How long is the lock-in period?",
        "shall constitute the lock-in period",
        ["eleven"],
        "lease",
    ),
    EvalCase(
        "How much is the security deposit?",
        "security deposit of Rs. 2,91,000",
        ["2,91,000"],
        "lease",
    ),
    EvalCase(
        "Within how many days must the security deposit be refunded?",
        "within twenty-one (21) days",
        ["twenty-one"],
        "lease",
    ),
    EvalCase(
        "By what percentage does the rent escalate?",
        "escalated by six per cent",
        ["six per cent"],
        "lease",
    ),
    EvalCase(
        "What are the monthly maintenance charges and who pays them?",
        "maintenance charges of Rs. 3,750",
        ["3,750"],
        "lease",
    ),
    EvalCase(
        "What notice period is required to terminate the lease?",
        "three (3) months' prior notice in writing",
        ["three (3) months"],
        "lease",
    ),
    EvalCase(
        "What is the carpet area of the flat?",
        "985 square feet of carpet area",
        ["985"],
        "lease",
    ),
    EvalCase(
        "What is the super built-up area?",
        "1,340 square feet of super built-up area",
        ["1,340"],
        "lease",
    ),
    EvalCase("Which car parking spaces are allotted?", "Nos. P-42 and P-43", ["P-42"], "lease"),
    EvalCase(
        "What interest applies to rent paid late?",
        "eighteen per cent (18%) per annum",
        ["eighteen per cent"],
        "lease",
    ),
    EvalCase(
        "How many people may ordinarily reside in the flat?",
        "shall not exceed five (5)",
        ["five"],
        "lease",
    ),
    EvalCase(
        "What must the lessee pay if the flat is not repainted?",
        "Rs. 22,000 towards repainting charges",
        ["22,000"],
        "lease",
    ),
    EvalCase("How much stamp duty was paid on the deed?", "stamp duty of Rs. 29,100", ["29,100"], "lease"),
    EvalCase(
        "Which courts have jurisdiction over disputes?",
        "courts at Bengaluru alone shall have jurisdiction",
        ["Bengaluru"],
        "lease",
    ),
    EvalCase(
        "Is the lessee allowed to keep pets?",
        "not more than one (1) domestic pet",
        ["one (1) domestic pet"],
        "lease",
    ),
    EvalCase(
        "What is the total term of the lease?",
        "term of thirty-three (33) months",
        ["thirty-three"],
        "lease",
    ),
    EvalCase(
        "What happens if the lessee vacates during the lock-in period?",
        "two (2) months' rent as liquidated damages",
        ["liquidated damages"],
        "lease",
    ),
]

STUDY_NOTES: list[EvalCase] = [
    EvalCase(
        "Define BCNF.",
        "the determinant X is a superkey of the relation",
        ["superkey"],
        "study",
    ),
    EvalCase(
        "Which isolation level still permits phantom reads?",
        "Repeatable Read Prevented Prevented Possible",
        ["Repeatable Read"],
        "study",
    ),
    EvalCase(
        "State the write-ahead logging rule.",
        "must reach stable storage before",
        ["stable storage"],
        "study",
    ),
    EvalCase(
        "What are the three passes of ARIES recovery?",
        "an analysis pass that",
        ["analysis"],
        "study",
    ),
    EvalCase(
        "What is a dirty read?",
        "reads data written by an uncommitted transaction",
        ["uncommitted"],
        "study",
    ),
    EvalCase(
        "State the transitivity axiom.",
        "Transitivity: if X -> Y and Y -> Z then X -> Z",
        ["X -> Z"],
        "study",
    ),
    EvalCase(
        "What is the lossless join test for a binary decomposition?",
        "is a superkey of R1 or a superkey of R2",
        ["superkey"],
        "study",
    ),
    EvalCase(
        "Which two-phase locking variant avoids cascading rollback?",
        "Strict 2PL",
        ["Strict 2PL"],
        "study",
    ),
    EvalCase(
        "In which deadlock prevention scheme does the older transaction wait?",
        "wait-die scheme",
        ["wait-die"],
        "study",
    ),
    EvalCase(
        "What is the examination weightage of this unit?",
        "22 marks",
        ["22 marks"],
        "study",
    ),
    EvalCase(
        "Which normal form removes transitive dependency?",
        "3NF 2NF and no transitive dependency",
        ["3NF"],
        "study",
    ),
    EvalCase(
        "What is the closure of an attribute set?",
        "the set of all attributes functionally determined by X",
        ["functionally determined"],
        "study",
    ),
]

OUT_OF_CORPUS: list[EvalCase] = [
    EvalCase("What is the capital of France?", category="out_of_corpus"),
    EvalCase("How do I train a convolutional neural network on ImageNet?", category="out_of_corpus"),
    EvalCase("What is the recipe for a chocolate sponge cake?", category="out_of_corpus"),
    EvalCase("Who won the 2018 FIFA World Cup final?", category="out_of_corpus"),
    EvalCase("What is the boiling point of mercury in kelvin?", category="out_of_corpus"),
    EvalCase("How do I install a Kubernetes cluster on Ubuntu?", category="out_of_corpus"),
    EvalCase("Summarise the plot of Hamlet.", category="out_of_corpus"),
    EvalCase("What were the quarterly earnings of Apple in 2024?", category="out_of_corpus"),
]


def all_cases(include_out_of_corpus: bool = True) -> list[EvalCase]:
    cases = [*CASE_LAW, *LEASE_DEED, *STUDY_NOTES]
    if include_out_of_corpus:
        cases += OUT_OF_CORPUS
    return cases


def in_corpus_cases() -> list[EvalCase]:
    return [*CASE_LAW, *LEASE_DEED, *STUDY_NOTES]
