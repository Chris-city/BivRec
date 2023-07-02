# BivRec
In order to facilitate the follow-up work, we provide five versions of the code, respectively:

Comi: Multi-interest recommendation model including ComiRec, MIND, and other models.
https://github.com/ShiningCosmos/pytorch_ComiRec.git

PureID/MM: Use interest-structured blocks to extract either ID or multimodal features for recommendations, which is a single-side version of BivRec.
Specifically, We also explored whether the interest-structured blocks can be stacked to connect larger models, where the information_passing function corresponds to four cases:
together: Interest tokens and historical sequences pass together through the common transformer layer for pre-information propagation.
self: The historical purchase sequence passes through the transformer layer separately
Dual: The historical purchase sequence passes through the transformer separately and then spliced together through a transformer.

PureID/MM ++: Adds a constraint to perform constraints based on PureID/MM.

BivRec-normal: Interest structured block is not used for multimodal feature extraction but uses some basic methods (e.g., concat, sum, attention) as supervisory signals for ID recommendation.

Bivrec-noi: use all blocks

The code is still being sorted out. We'll put all five versions together as soon as possible.
