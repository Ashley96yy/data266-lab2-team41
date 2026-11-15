# Part 3 quality revision: development protocol

The earlier Windows evaluation remains intact under .runtime/eval and its
individual trace directories. The unchanged 15 primary scenarios, 10 red-team
cases and all private inputs remain frozen. This iteration fixes implementation
defects found from code and the prior traces; it is a development rerun, not a
claim of a fresh independent holdout. Human pass/fail and critic labels remain
blank until two real members independently review the results.

Changes fixed before this rerun: compile the Prompt Agent structured output
under the diffusion text budget without clipping the user request; separate
quality and safety schemas with one bounded format retry; record CPU CLIP
image/prompt cosine similarity without a fabricated passing threshold; pass
shared dataclass state through five stages and log timed input/output events;
ask at least two visual questions by default; use geometry-preserving
photometric enhancement rather than resynthesizing an enhancement request.
Enhancement has zero diffusion steps and is explicitly identified as a Pillow
backend; the unchanged SD safety checker and VLM safety review remain enabled.
Bounded revision strength applies the original approved enhancement operation
again to the original source. Style/domain generation still uses SD 1.5 and
the pinned team LoRA, with adapter deactivation checked afterward.

A new synthetic diagnostic drawing is generated separately from the fixed
fixtures. Real Vision, base generation, team LoRA, enhancement, safety checks,
CLIP and nested VLM critique must execute before the frozen rerun. Every
mode writes to a new timestamped .runtime/part3-revisions directory. Preserve
failed attempts and raw schema failures. Shared first-generation prompts,
settings, model revisions and seeds are checked between the two modes.

Do not install or replace CUDA/PyTorch. Reuse the existing environment and
model caches. All outputs and portraits remain private on this computer.
No GPU performance, quality score, attack-success rate or human agreement is
claimed until the corresponding evidence has actually been generated.

Visual inspection of the first two development reruns exposed weak medium
conversion and retained color in a requested black-and-white result. They
remain archived, including critic false accepts. The next development version
uses a bounded Qwen Prompt Agent to write descriptive diffusion captions,
validated for explicit requested colors/counts and the real CLIP token budget.
Style img2img strength is 0.70; domain strength is 0.55 with team LoRA weight
0.65. Those settings are fixed equally across both critique conditions before
the rerun. Explicit monochrome requests receive a recorded grayscale operation,
followed by another safety check; the critic also checks remaining chromatic
pixels as a quality constraint. These are implementation changes informed by
observed development failures, not untouched-holdout improvements. No automatic
style, CLIP or completion result substitutes for the two members' judgments.

A subsequent development run exposed an overly strict explicit-single validator:
Qwen used an indefinite article for one subject. Normalization now accepts that
representation and adds a single-subject cue, while still validating requested
colors/counts and token limits. Approved-medium material cues are prefixed by
the Prompt Agent; no fixture answers or rubrics enter generation. Format retries
include actual validation feedback. Every feedback merge reenters the timed Prompt
stage. Failed notebook/validator runs remain intact; no outcome is relabeled.
