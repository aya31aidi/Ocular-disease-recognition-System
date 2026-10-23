"""Reference content for the eight classes the model predicts.

The order of CONDITIONS matters: it mirrors the integer labels assigned in the
training notebook (Normal=0, Cataract=1, Diabetes=2, ... Other=7), so index i
of the softmax output maps to CONDITIONS[i].
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Condition:
    key: str
    name: str
    short_name: str
    tagline: str
    overview: str
    fundus_signs: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    urgency: str = "routine"  # routine | soon | prompt
    training_images: int = 0


URGENCY_LABELS = {
    "routine": "Routine follow-up",
    "soon": "Book an eye exam soon",
    "prompt": "Seek care promptly",
}

CONDITIONS: list[Condition] = [
    Condition(
        key="normal",
        name="Normal fundus",
        short_name="Normal",
        tagline="No signs of the seven conditions in scope.",
        overview=(
            "The retina, optic disc, macula and blood vessels look within normal "
            "limits for the patterns the model has learned. A normal screening result "
            "does not rule out conditions outside the model's eight classes, or early "
            "changes too subtle to show on a single photograph."
        ),
        fundus_signs=[
            "Clear view of the retina with well-defined vessels",
            "Optic disc with a healthy pink rim and a small central cup",
            "Even reflectance around the macula",
        ],
        symptoms=["None expected"],
        next_steps=[
            "Keep to the eye-exam schedule a clinician recommends for the patient's age and risk factors",
            "Report any new change in vision, even with a normal result",
        ],
        urgency="routine",
        training_images=700,
    ),
    Condition(
        key="cataract",
        name="Cataract",
        short_name="Cataract",
        tagline="Clouding of the eye's natural lens.",
        overview=(
            "A cataract is a clouding of the lens behind the iris. It develops slowly with "
            "age, and also with diabetes, smoking, UV exposure or long-term steroid use. "
            "On a fundus photograph the cataract itself is not visible; what shows up is a "
            "hazy, low-contrast view of the retina because the cloudy lens scatters light."
        ),
        fundus_signs=[
            "Overall blur and loss of contrast across the image",
            "Fine vessels fading out or becoming hard to trace",
            "Washed-out or uneven illumination",
        ],
        symptoms=[
            "Cloudy or blurred vision",
            "Glare and halos around lights, especially at night",
            "Colours looking faded or yellowed",
            "Frequent changes in glasses prescription",
        ],
        next_steps=[
            "Comprehensive eye exam with a slit-lamp assessment",
            "Cataract surgery (lens replacement) once vision limits daily activities",
        ],
        urgency="routine",
        training_images=594,
    ),
    Condition(
        key="diabetic_retinopathy",
        name="Diabetic retinopathy",
        short_name="Diabetic retinopathy",
        tagline="Retinal vessel damage caused by diabetes.",
        overview=(
            "Long periods of high blood glucose weaken the small blood vessels of the "
            "retina. They leak, bleed and, in later stages, are replaced by fragile new "
            "vessels. It is one of the leading causes of preventable blindness in working-age "
            "adults, and often has no symptoms until it is advanced, which is why annual "
            "screening matters. The training images for this class were labelled "
            "non-proliferative diabetic retinopathy."
        ),
        fundus_signs=[
            "Microaneurysms: tiny red dots along the capillaries",
            "Dot and blot haemorrhages",
            "Hard exudates: sharp yellow lipid deposits",
            "Cotton-wool spots: fluffy white patches of damaged nerve fibres",
        ],
        symptoms=[
            "Usually none in early stages",
            "Floaters or dark spots",
            "Blurred or fluctuating vision",
        ],
        next_steps=[
            "Dilated eye exam with an ophthalmologist to grade severity",
            "Tighter control of blood glucose, blood pressure and cholesterol",
            "Treatment for advanced disease may include anti-VEGF injections or laser",
        ],
        urgency="soon",
        training_images=1818,
    ),
    Condition(
        key="glaucoma",
        name="Glaucoma",
        short_name="Glaucoma",
        tagline="Progressive damage to the optic nerve.",
        overview=(
            "Glaucoma is a group of diseases that damage the optic nerve, often linked to "
            "raised pressure inside the eye. Vision loss starts at the edges of the visual "
            "field and is permanent, but treatment can slow or stop it. Most people with "
            "the common open-angle form notice nothing until the disease is advanced."
        ),
        fundus_signs=[
            "Enlarged cup-to-disc ratio",
            "Thinning or notching of the neuroretinal rim",
            "Small haemorrhages at the edge of the optic disc",
        ],
        symptoms=[
            "Often none early on",
            "Gradual loss of peripheral (side) vision",
            "Sudden eye pain, redness, halos and nausea in acute angle-closure, which is an emergency",
        ],
        next_steps=[
            "Eye-pressure measurement, visual-field test and OCT of the optic nerve",
            "Treatment with pressure-lowering eye drops, laser or surgery",
        ],
        urgency="soon",
        training_images=610,
    ),
    Condition(
        key="hypertension",
        name="Hypertensive retinopathy",
        short_name="Hypertension",
        tagline="Retinal changes caused by high blood pressure.",
        overview=(
            "Sustained high blood pressure narrows and stiffens the retinal arteries. "
            "Because the retina is the only place where blood vessels can be seen directly, "
            "these changes are often a window into the state of vessels elsewhere in the "
            "body, including the heart, kidneys and brain."
        ),
        fundus_signs=[
            "Narrowed arterioles and arteriovenous nicking",
            "Flame-shaped haemorrhages",
            "Cotton-wool spots",
            "Swelling of the optic disc in severe cases",
        ],
        symptoms=[
            "Usually none",
            "Headaches or blurred vision with very high blood pressure",
        ],
        next_steps=[
            "Blood-pressure check and management with a primary care physician",
            "Eye exam to grade the retinopathy",
            "Very high blood pressure together with vision changes needs emergency care",
        ],
        urgency="soon",
        training_images=374,
    ),
    Condition(
        key="myopia",
        name="Pathological myopia",
        short_name="Myopia",
        tagline="Severe short-sightedness with structural changes.",
        overview=(
            "In high myopia the eyeball grows too long, stretching the retina and the layers "
            "beneath it. Beyond blurry distance vision, the stretching raises the risk of "
            "retinal detachment, myopic maculopathy and glaucoma later in life."
        ),
        fundus_signs=[
            "Tilted optic disc with a crescent of atrophy beside it",
            "Tessellated (tigroid) fundus where choroidal vessels show through",
            "Thinning and patchy atrophy around the macula",
        ],
        symptoms=[
            "Blurred distance vision",
            "Distorted central vision if the macula is affected",
            "Sudden flashes, a shower of floaters or a shadow across vision can signal detachment",
        ],
        next_steps=[
            "Refraction and dilated retinal exam",
            "Regular monitoring of the macula and peripheral retina",
            "Urgent review for any sudden flashes, floaters or curtain-like shadow",
        ],
        urgency="routine",
        training_images=479,
    ),
    Condition(
        key="amd",
        name="Age-related macular degeneration",
        short_name="AMD",
        tagline="Deterioration of the central retina with age.",
        overview=(
            "AMD affects the macula, the small area responsible for sharp central vision "
            "used for reading and recognising faces. The dry form progresses slowly; the wet "
            "form, where abnormal vessels leak under the retina, can damage vision within "
            "weeks and needs fast treatment."
        ),
        fundus_signs=[
            "Medium or large drusen clustered at the macula",
            "Pigment clumping or loss",
            "Geographic atrophy, or haemorrhage and fluid in the wet form",
        ],
        symptoms=[
            "Blurred or dim central vision",
            "Straight lines looking wavy",
            "Difficulty reading or recognising faces",
        ],
        next_steps=[
            "Retinal exam with OCT imaging",
            "Home monitoring with an Amsler grid",
            "AREDS2 supplements for intermediate disease; anti-VEGF injections for wet AMD",
        ],
        urgency="soon",
        training_images=551,
    ),
    Condition(
        key="other",
        name="Other abnormality",
        short_name="Other",
        tagline="Findings outside the six named diseases.",
        overview=(
            "This class covers abnormal findings that do not fit the other categories. In "
            "this project's training data it was built from images labelled with drusen: "
            "yellow deposits under the retina. A few small drusen are common with age, "
            "while larger or numerous ones are linked to a higher risk of AMD."
        ),
        fundus_signs=[
            "Scattered yellow-white spots",
            "Irregular patches that do not match a typical disease pattern",
        ],
        symptoms=["Varies with the underlying cause; often none"],
        next_steps=[
            "Eye exam to characterise the finding",
            "Follow-up imaging to see whether it changes over time",
        ],
        urgency="routine",
        training_images=324,
    ),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}
CLASS_KEYS = [c.key for c in CONDITIONS]
