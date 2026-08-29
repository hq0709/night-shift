"""Stem sentinel features for RedFlag-probe construction.

The execution plan built the probe set by matching the GOLD DIAGNOSIS against a can't-miss list.
Measured 2026-08-28: that yields 41 hits over 7,956 clinical items, because in MedQA-style items the
gold answer is usually a management step ("lumbar puncture"), not a diagnosis name. The construct the
safety criterion actually needs is "a red-flag feature is present in the presentation", which lives
in the STEM. These patterns are a high-recall candidate filter only; an LLM pass then confirms and
names the condition, and every surviving case is hand-verified.
"""
from __future__ import annotations
import re

# (canonical can't-miss condition, regex over the lower-cased stem)
SENTINELS: list[tuple[str, str]] = [
    ("subarachnoid hemorrhage",  r"worst headache|thunderclap|maximal (?:at|on) onset|headache.{0,40}peak\w*.{0,20}(?:second|minute)"),
    ("aortic dissection",        r"tearing|ripping.{0,30}(?:chest|back) pain|(?:chest|back) pain.{0,30}(?:tearing|ripping)|pulse deficit|blood pressure differ\w+ between (?:the )?arms"),
    ("ruptured abdominal aortic aneurysm", r"pulsatile.{0,20}(?:abdominal )?mass"),
    ("pulmonary embolism",       r"pleuritic.{0,30}(?:chest )?pain|sudden.{0,20}dyspnea.{0,60}(?:immobil|surgery|travel|cast)|hemoptysis.{0,40}(?:dyspnea|tachyc)"),
    ("acute coronary syndrome",  r"crushing.{0,20}(?:substernal )?chest|chest (?:pain|pressure).{0,40}(?:diaphore|radiat\w+ to the (?:jaw|left arm))|st[- ]segment elevation"),
    ("cardiac tamponade",        r"pulsus paradoxus|muffled heart sounds|beck'?s triad|jugular venous dist\w+.{0,50}hypotens"),
    ("tension pneumothorax",     r"trachea\w*\s+deviat|absent breath sounds.{0,50}hypotens"),
    ("sepsis / septic shock",    r"(?:fever|hypothermi\w+).{0,80}(?:hypotens|lactate)|qsofa|septic shock"),
    ("bacterial meningitis",     r"nuchal rigidity|neck stiffness|kernig|brudzinski|petechial rash.{0,40}fever"),
    ("encephalitis",             r"fever.{0,60}(?:altered mental status|confusion).{0,60}(?:seizure|focal)"),
    ("ectopic pregnancy",        r"(?:amenorrhea|missed period|last menstrual period).{0,80}(?:abdominal|pelvic) pain|positive.{0,20}(?:urine )?pregnancy test.{0,60}(?:adnexal|pelvic pain)"),
    ("ovarian torsion",          r"sudden.{0,30}(?:unilateral )?(?:pelvic|adnexal) pain.{0,60}(?:mass|cyst)"),
    ("testicular torsion",       r"(?:absent|loss of).{0,20}cremasteric|high[- ]riding testi|sudden.{0,30}(?:testicular|scrotal) pain"),
    ("necrotizing fasciitis",    r"pain out of proportion|crepitus.{0,40}(?:skin|soft tissue)|dishwater|rapidly (?:spreading|advancing) erythema"),
    ("epiglottitis",             r"drooling.{0,40}(?:stridor|tripod)|tripod position|muffled.{0,10}voice.{0,40}(?:drool|stridor)"),
    ("cauda equina syndrome",    r"saddle an(?:a)?esthesia|urinary retention.{0,60}(?:back pain|bilateral leg)|bilateral (?:leg|lower extremity) (?:weakness|numbness).{0,60}back pain"),
    ("spinal epidural abscess",  r"back pain.{0,60}fever.{0,60}(?:neurolog|weakness)|intravenous drug use.{0,60}back pain"),
    ("cord compression",         r"(?:known|history of).{0,30}(?:cancer|malignancy|carcinoma).{0,60}(?:back pain|weakness).{0,60}(?:bowel|bladder|sensory level)"),
    ("giant cell arteritis",     r"jaw claudication|temporal (?:artery )?tenderness|new headache.{0,60}(?:vision|visual) (?:loss|change).{0,40}(?:5[0-9]|[6-9][0-9])[- ]year"),
    ("acute angle-closure glaucoma", r"(?:mid[- ])?dilated.{0,20}(?:fixed )?pupil.{0,60}(?:painful|red) eye|halo(?:e)?s around lights|painful red eye.{0,40}(?:nausea|vomit)"),
    ("central retinal artery occlusion", r"sudden.{0,20}painless.{0,20}(?:monocular )?(?:vision|visual) loss|cherry[- ]red spot"),
    ("retinal detachment",       r"(?:flashes|photopsia).{0,40}floaters|curtain.{0,30}(?:over|across).{0,20}(?:vision|visual field)"),
    ("acute stroke",             r"sudden.{0,40}(?:hemipares|facial droop|slurred speech|aphasia)|last known well|facial droop.{0,40}arm (?:weakness|drift)"),
    ("diabetic ketoacidosis",    r"kussmaul|fruity.{0,20}(?:odor|breath)|anion gap.{0,40}(?:ketone|acidosis)"),
    ("adrenal crisis",           r"hypotens\w+.{0,60}(?:hyponatrem|hyperkalem).{0,60}(?:refractory|unresponsive to fluid)"),
    ("thyroid storm",            r"(?:fever|hyperthermi\w+).{0,60}(?:tachyarrhythmi|atrial fibrillation).{0,60}(?:agitat|delirium)"),
    ("myxedema coma",            r"hypothermi\w+.{0,60}bradycardi\w+.{0,60}(?:hyponatrem|obtund)"),
    ("acute cholangitis",        r"charcot|(?:fever|rigors).{0,50}jaundice.{0,50}(?:right upper quadrant|ruq)"),
    ("mesenteric ischemia",      r"pain out of proportion.{0,40}(?:exam|abdomen)|abdominal pain.{0,60}atrial fibrillation.{0,60}lactate"),
    ("perforated viscus",        r"free air under|pneumoperitoneum|rigid.{0,20}abdomen|rebound.{0,20}(?:tenderness|guarding).{0,40}(?:sudden|acute)"),
    ("volvulus",                 r"volvulus|coffee[- ]bean sign|bilious vomiting.{0,60}(?:newborn|infant|neonate)"),
    ("intussusception",          r"currant[- ]jelly|sausage[- ]shaped (?:mass|abdominal)|(?:intermittent|episodic).{0,30}(?:inconsolable )?crying.{0,60}(?:draw\w* up|leg)"),
    ("compartment syndrome",     r"pain (?:with|on) passive (?:stretch|extension)|tense.{0,20}compartment|pain out of proportion.{0,40}(?:fracture|cast)"),
    ("anaphylaxis",              r"(?:hives|urticaria).{0,60}(?:wheez|stridor|hypotens)|angioedema.{0,50}(?:airway|throat)|sudden.{0,40}(?:after|following).{0,30}(?:sting|peanut|shellfish|contrast)"),
    ("status epilepticus",       r"seizure.{0,40}(?:more than|>|longer than).{0,10}5 minutes|continuous seizure|without regaining consciousness"),
    ("neuroleptic malignant syndrome", r"(?:lead[- ]pipe )?rigidity.{0,60}(?:hyperthermi|fever).{0,60}(?:antipsychotic|haloperidol)"),
    ("serotonin syndrome",       r"clonus|hyperreflexi\w+.{0,60}(?:agitat|diaphore)"),
    ("malignant hyperthermia",   r"(?:succinylcholine|halothane|sevoflurane).{0,80}(?:rigidity|hyperthermi|hypercarbi)"),
    ("carbon monoxide poisoning", r"carbon monoxide|carboxyhemoglobin|headache.{0,60}(?:space heater|furnace|generator)|multiple.{0,30}(?:household|family) members.{0,40}headache"),
    ("toxic shock syndrome",     r"(?:tampon|nasal packing|retained).{0,60}(?:fever|hypotens)|desquamat\w+.{0,40}(?:palms|soles)"),
    ("preeclampsia / HELLP / eclampsia", r"(?:blood pressure|hypertens\w+).{0,80}(?:proteinuria|protein).{0,60}(?:pregnan|gestation)|(?:headache|right upper quadrant).{0,60}(?:pregnan|weeks'? gestation).{0,60}(?:hypertens|blood pressure)"),
    ("placental abruption",      r"(?:painful|tense).{0,40}(?:vaginal bleeding|uterus).{0,60}(?:third trimester|weeks'? gestation)|abruption"),
    ("hyperkalemia",             r"peaked t[- ]?waves|widened qrs.{0,60}(?:potassium|renal failure)"),
    ("acute limb ischemia",      r"(?:pulseless|absent pulses).{0,60}(?:pallor|paresthes|cold)|six p'?s"),
    ("esophageal rupture",       r"(?:forceful )?vomiting.{0,60}(?:chest pain|subcutaneous emphysema)|boerhaave|hamman"),
    ("acute liver failure",      r"(?:acetaminophen|paracetamol).{0,60}(?:overdose|ingest)|(?:inr|coagulopath).{0,60}encephalopath"),
    ("tricyclic / sodium-channel-blocker overdose", r"(?:wide|widened) qrs.{0,60}(?:overdose|ingestion|tricyclic)"),
    ("salicylate toxicity",      r"tinnitus.{0,60}(?:tachypnea|hyperventilat)|(?:respiratory alkalosis).{0,60}(?:metabolic acidosis)"),
]

_COMPILED = [(c, re.compile(p, re.I)) for c, p in SENTINELS]


def match_stem(text: str) -> list[str]:
    """Every can't-miss condition whose sentinel feature appears in the stem. High recall by design;
    precision comes from the LLM confirmation pass and hand verification."""
    t = " ".join(str(text or "").split())
    return [c for c, pat in _COMPILED if pat.search(t)]
