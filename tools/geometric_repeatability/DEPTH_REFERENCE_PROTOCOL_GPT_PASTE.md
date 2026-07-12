```text
Please review the attached protocol package for a proposed new geometry-evidence experiment.

Main file:
- DEPTH_REFERENCE_PROTOCOL.md

Context:
- We already have a strict odd/even repeatability evaluator, but expanding that evaluator to every baseline/scene requires expensive retraining.
- We now want a second geometry-evidence protocol that does not require retraining all baselines.
- The specific failure mode we want to quantify is the one we observe by inspection: roofs/facades/floating structure expanding toward the camera in held-out renders.

Please judge whether the proposed protocol is scientifically clean and top-conference-appropriate enough to implement.

Please answer these questions concretely:
1. Is the protocol acceptable if we explicitly describe it as reference-depth-based geometric evaluation rather than absolute GT depth accuracy?
2. Do you agree with the pipeline:
   training-only RGB views -> external MVS reference geometry -> reference mesh -> held-out reference depth D_ref
   while each trained method renders held-out thermal depth D_model in the same cameras?
3. Is FrontIntrusionRate@delta a strong primary metric for the observed “geometry growing toward the camera” failure mode?
4. Is the threshold sweep 0.10 / 0.25 / 0.50 / 1.00 m reasonable for large outdoor scenes?
5. Do you see any blocker that should be fixed before implementation?
6. If you approve implementation, do you agree that the first pilot should be Building + 5 methods + T models only?

Important wording constraint:
- acceptable wording: reference-depth-based geometric evaluation / held-out geometry consistency against a training-only MVS reference
- unacceptable wording: ground-truth depth accuracy / absolute geometry accuracy
```
