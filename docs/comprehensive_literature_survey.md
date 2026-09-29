# Literature survey: visually provoked seizures and video content

Last reviewed: 2026-09-29

## Scope and scientific caution

This survey asks what published research and major accessibility standards say
about visual content associated with visually provoked seizures, how software
detects that content, and how video might be modified. It is not a clinical
risk model, medical advice, or evidence that this repository can make a video
"seizure-safe." The response to visual stimuli varies among people, displays,
viewing environments, and clinical populations. Passing a computational test
does not establish zero risk.

The search covered clinical reviews, experimental stimulus studies, standards,
computer-vision detection and mitigation papers, deflickering and frame
interpolation research, public test corpora, and open implementations. Sources
were prioritized in this order: peer-reviewed reviews and consensus reports;
official standards; peer-reviewed algorithms; preprints; and vendor or project
documentation. This is a broad, reproducible literature map, not a claim to
have indexed every publication on the internet.

## Main conclusions

1. **There is no single "epilepsy-inducing video" signature.** Relevant
   dimensions include flash rate, light-dark contrast or luminance change,
   affected visual area, the same pixels undergoing opposing transitions,
   saturated chromatic transitions (especially red), and repetitive spatial
   patterns. Individual susceptibility and viewing conditions also matter.
2. **Sensitivity is frequency-dependent but not confined to one frequency.**
   Reviews report a broad response range, with the largest population response
   commonly around 15--25 Hz. Standards therefore analyze a range rather than
   searching for one exact frequency.
3. **A flash is not equivalent to arbitrary inter-frame difference.** Standards
   and conformance research pair sufficiently large transitions of opposing
   polarity and require spatial overlap. Motion, a single cut, or a monotonic
   fade can produce a large frame difference without constituting repeated
   flashing in this sense.
4. **Area matters.** Whole-frame averaging can hide a small intense region,
   while a single changed-pixel count can overreact to camera motion. Spatial
   location, overlap, and visual-angle or screen-area rules are essential.
5. **Static or moving stripes are a separate risk category.** A temporal
   luminance detector alone cannot cover high-contrast regular patterns.
6. **Display assumptions matter.** Normalized RGB differences omit absolute
   display luminance, HDR behavior, viewing distance, field of view, transfer
   functions, and colorimetry. Such a metric remains a useful computational
   proxy, but it is not standards conformance.
7. **The evidence for automatic mitigation is much smaller than the evidence
   for characterization and detection.** One early adaptive-filter study tested
   11 photosensitive patients on a particular television sequence. Modern
   open-source and learned systems are promising, but broad independent
   clinical validation is not established by their publications or READMEs.
8. **Blindly combining filters is not supported.** Detection, classification,
   mitigation, and validation should be separate modules, and each added
   operation should earn its place through an ablation and perceptual-quality
   evaluation.

## What the clinical literature says

The 2005 Epilepsy Foundation review by Fisher et al. remains a foundational
synthesis. It describes photosensitivity as uncommon in the overall population
but more prevalent among people with epilepsy, with higher prevalence in young
people and particular epilepsy syndromes. It reports a broad effective flash
frequency range, a stronger population response near 15--25 flashes per second,
and important roles for contrast, retinal area, patterns, and red light.

The Foundation's 2022 update expands the content context to social video,
games, films, and immersive displays. It identifies frequency, brightness,
area, red flashes, and oscillating stripes as notable stimulus variables. The
2026 consensus paper is the newest expert synthesis located in this search and
should be treated as the current clinical starting point rather than replacing
standards with an improvised threshold.

Experimental studies add nuance:

- Long-wavelength red stimulation has been unusually provocative in some
  photosensitive participants; this motivates a chromatic-flash analyzer in
  addition to a luminance analyzer.
- Pattern studies and guidelines show that regular, high-contrast stripes can
  be provocative independently of whole-frame flashing.
- Video-game case series describe both photic mechanisms and non-photic
  contributors such as fatigue, stress, and prolonged play. Pixel analysis
  cannot model those person-level factors.
- The 1997 Pokemon broadcast is a prominent mass-event case, but it should not
  be generalized into a universal content threshold without the subsequent
  clinical and standards literature.

## What standards actually test

Major rules are related but not identical. WCAG, ITU-R BT.1702, ISO 9241-391,
and national broadcasting guidance differ in definitions, geometry, and
application. A recent gap analysis found that the standards are not fully
harmonized and that HDR, high-frame-rate displays, and immersive viewing expose
important gaps.

WCAG 2.2 Success Criterion 2.3.1 uses a simple top-level rule: content should
not contain more than three flashes in any one-second period unless it remains
below the general-flash and red-flash thresholds. The detailed test incorporates
relative luminance, transition direction, spatial area, and viewing assumptions.
"Three flashes" is therefore not a complete implementation specification by
itself.

ITU-R BT.1702 is broadcasting-oriented and includes guidance for flashes and
regular patterns. ISO 9241-391 addresses reduction of photosensitive-seizure
risk in electronic visual displays. These standards should be implemented from
their normative text and tested against ground-truth media; they should not be
approximated by tuning the project's normalized threshold until the output
looks different.

## Detection research

### Standards-oriented detection

- **Carreira et al. (2015)** present real-time detection of flashing video
  content guided by ITU-R BT.1702. The important design lesson is explicit
  spatiotemporal analysis rather than only a scalar mean over the frame.
- **Alzubaidi et al. (2016)** present a parallel real-time spatiotemporal
  detection scheme and controlled synthetic patterns. This supports both
  streaming operation and deterministic boundary-case generation.
- **Jordan (2025)** evaluates conformance testing itself. Particularly relevant
  cases include irregular timing, multistep transitions, spatial overlap at the
  same pixels, and out-of-sync regional changes. The paper explains why a flash
  must be constructed from a pair of opposing transitions rather than a single
  absolute difference.
- **South, Saffo, and Borkin (2021)** study detection and user-facing defenses
  for seizure-inducing GIFs on social media. It is valuable because it treats
  the interface and user's ability to avoid or control media as part of the
  system, not merely the signal-processing kernel.

### Learned transformation

Barbu, Banda, and Katz (2020) formulate accessibility-oriented video-to-video
transformation using a stacked ConvLSTM U-Net, synthetic augmentation, and a
differentiable visual model. It demonstrates that a learned sequence model can
optimize temporal transformation, but it also introduces training-distribution,
interpretability, and validation questions. Descriptions such as "harmless" in
an individual paper are authors' experimental conclusions, not a certification
that an unrelated implementation or arbitrary output is clinically safe.

### Open software

Apple's VideoFlashingReduction project, EA's IRIS, FFmpeg's photosensitivity
filter, EPI-LENS, and the TRACE test-media repository are useful engineering
references. They are not interchangeable certifications. IRIS explicitly
states that it is not a substitute for third-party testing, and recent
conformance work warns that publicly available tools may implement different
rules or incomplete approximations.

## Mitigation research

Nomura et al. (2000) describe an adaptive temporal filter designed to attenuate
roughly 10--30 Hz activity and report testing with 11 photosensitive patients
using a Pokemon sequence. It is unusually relevant because it connects a
filter with a clinical response measurement, but the sample, stimulus, display,
and algorithm are specific. It does not validate this repository's local
blending, luminance clamping, or frame-generation implementations.

General video deflickering methods optimize temporal consistency and visual
quality, sometimes using optical flow or learned atlases. Frame interpolation
methods such as RIFE and FILM reconstruct plausible intermediate motion. These
are adjacent technologies, not PSE risk models. They can create occlusion,
warping, or hallucination artifacts and can preserve a sustained alternating
signal if the sequence objective does not explicitly penalize it. This agrees
with the negative result from this repository's triplet-based frame-generation
experiment.

## Implications for this repository

The current signed-polarity reversal detector is a better research direction
than the original absolute frame-difference threshold because it distinguishes
an opposing transition pair from a scene cut or monotonic fade. It is still not
a standards-conformant analyzer.

A defensible next architecture is:

1. **Input characterization:** preserve source resolution and timestamps;
   record transfer function, color primaries, frame rate, and HDR metadata when
   available.
2. **Separate analyzers:**
   - paired general-luminance transitions with same-pixel overlap;
   - saturated-red or chromatic transitions in an explicitly documented color
     space;
   - regular high-contrast spatial patterns and their motion;
   - scene-cut, camera-motion, fade, and moving-object confound labels.
3. **Temporal decision:** count qualifying opposing pairs in timestamp-based
   one-second windows, not a fixed number of frames, and retain irregular and
   multistep transitions.
4. **Spatial decision:** use localized connected regions or multiscale tiles,
   preserve overlap information, and report both pixel area and assumed visual
   angle. Do not hide small intense regions in a global mean.
5. **Mitigation candidates:** compare adaptive local luminance compression,
   motion-aligned temporal optimization, and carefully feathered local blending.
   Avoid independently generating every triplet in a continuous alternating
   run.
6. **Validation:** test exact boundary cases with TRACE ground-truth media,
   deterministic synthetic cases, and cross-tool comparisons against IRIS,
   Apple's implementation, and FFmpeg. Disagreements must be investigated, not
   decided by majority vote.
7. **Evaluation:** jointly report residual activity, localization precision and
   recall, modification area, PSNR/SSIM, temporal perceptual artifacts, runtime,
   and unchanged false-positive controls. If people are involved, obtain proper
   ethical and clinical oversight before exposing photosensitive participants
   to test stimuli.

## High-priority research gaps

- Public, standards-labeled corpora remain limited; irregular and adversarial
  boundary cases deserve much more coverage.
- HDR and wide-color-gamut content break assumptions inherited from SDR
  normalized RGB.
- Viewing distance and field of view make a pixel-area fraction an incomplete
  description, especially for phones and head-mounted displays.
- Algorithms need better separation of flashing from edits, camera motion,
  object motion, and ordinary cinematic lighting.
- Mitigation quality needs sequence-level perceptual evaluation; per-frame
  PSNR and SSIM miss ghosting, judder, and phase artifacts.
- Independent clinical validation of automatic transformations is sparse.
- A transparent detector can be standards-inspired and scientifically useful
  without claiming that it predicts whether a particular person will have a
  seizure.

## Selected bibliography and resources

### Clinical reviews, consensus, and stimulus studies

- Fisher et al. (2005), *Photic- and Pattern-induced Seizures: A Review for the
  Epilepsy Foundation of America Working Group*. Epilepsia.
  DOI: 10.1111/j.1528-1167.2005.31405.x.
- Fisher et al. (2022), *Visually sensitive seizures: An updated review by the
  Epilepsy Foundation*. Epilepsia. PMID: 35132632.
- Fisher et al. (2026), *Visually-provoked seizures: Consensus of the Epilepsy
  Foundation Working Group*. Epilepsia. DOI: 10.1111/epi.18702.
- Harding and Harding (2010), *Photosensitive epilepsy and image safety*.
  Applied Ergonomics. DOI: 10.1016/j.apergo.2008.08.005.
- Binnie et al. (2002), *Characterizing the Flashing Television Images that
  Precipitate Seizures*. SMPTE Journal.
- Wilkins, Emmett, and Harding (2005), *Characterizing the patterned images
  that precipitate seizures and optimizing guidelines to prevent them*.
  Epilepsia.
- Porciatti et al. (2000), *Lack of cortical contrast gain control in human
  photosensitive epilepsy*. Nature Neuroscience.
- Takahashi et al. (1995), *Wavelength dependency of photoparoxysmal responses
  in photosensitive patients*. Epilepsia. DOI:
  10.1111/j.1528-1157.1995.tb00466.x.
- Takahashi et al. (1997), *Wavelength dependency of photoparoxysmal responses
  in photosensitive epilepsy*. Tohoku Journal of Experimental Medicine.
  DOI: 10.1620/tjem.181.311.
- Kasteleijn-Nolst Trenite et al. (1994), *Video game induced seizures*.
  Journal of Neurology, Neurosurgery & Psychiatry. PMID: 8057115.

### Detection and transformation

- Nomura et al. (2000), *A New Adaptive Temporal Filter: Application to
  Photosensitive Seizure Patients*. DOI: 10.1046/j.1440-1819.2000.00770.x.
- Carreira et al. (2015), *Automatic Detection of Flashing Video Content*.
  DOI: 10.1109/QoMEX.2015.7148104.
- Alzubaidi et al. (2016), *Parallel Scheme for Real-Time Detection of
  Photosensitive Seizures*. DOI: 10.1016/j.compbiomed.2016.01.008.
- Barbu, Banda, and Katz (2020), *Deep video-to-video transformations for
  accessibility with an application to photosensitivity*. DOI:
  10.1016/j.patrec.2019.01.019.
- South, Saffo, and Borkin (2021), *Detecting and Defending Against
  Seizure-Inducing GIFs in Social Media*. CHI 2021.
- Jordan (2025), *Evaluating Conformance of Video Safety Tools for
  Photosensitive Epilepsy*. DOI: 10.1007/978-3-031-93848-1_7.

### Standards and gap analysis

- ITU-R BT.1702-3 (2023), *Guidance for the reduction of photosensitive
  epileptic seizures caused by television*.
- ISO 9241-391:2016, *Requirements, analysis and compliance test methods for
  the reduction of photosensitive seizures*.
- W3C, WCAG 2.2, Success Criterion 2.3.1 and its supporting explanation.
- Jordan and Vanderheiden, *International Guidelines for Photosensitive
  Epilepsy: Gap Analysis and Recommendations*.

### Public implementations and test media

- Apple, `VideoFlashingReduction` and its technical report.
- Electronic Arts, `IRIS` photosensitivity analysis tool.
- FFmpeg, `photosensitivity` video filter.
- TRACE RERC, `pse-test-media` ground-truth test cases.
- EPI-LENS, open photosensitivity analysis project.

The corresponding machine-readable references used by the report are in
`paper/references.bib`. Links to primary sources are maintained in the project
README and the references file where licensing permits.
