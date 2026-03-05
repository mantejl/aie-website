"use client"
import { useState, useEffect } from "react"

const tracks = [
    {
        name: "Sustainability",
        description:
            "Address AI's growing environmental footprint — from e-waste and energy consumption to freshwater scarcity. Develop frameworks that make AI computation more sustainable.",
        prompts: [
            "Training a single large-scale AI model can emit five times the lifetime emissions of a car, while data centers consume billions of gallons of water for cooling. How can we develop sustainable AI frameworks that prioritize energy-efficient architectures or utilize hardware-software co-design to minimize the environmental footprint of computation? Your solution should address the trade-off between model performance and ecological cost.",
            "Create an AI workload management system that accounts for regional water scarcity and cooling requirements when allocating computational resources across data centers. Your design should balance performance, latency, and ecological impact, incorporating environmental context beyond carbon emissions alone. How can AI infrastructure adapt to local environmental constraints while remaining globally scalable and efficient?",
            "Recent studies show wide scale AI usage is estimated to increase e-waste production by 1.2 to 5 million tons. Simultaneously the hardware needs for AI models are drastically increasing and accelerating the obsolescence of older generation GPUs. Create a solution which seeks to address e-waste ramifications within our environment, decrease the production of e-waste by AI models, or both.",
            "Illegal mining, industrial discharge, and overconsumption of fresh water resources has put our population at risk of losing continued access to fresh water resources at an affordable rate. How can we design a system that may detect heavy metal in freshwater systems in real time and optimizes intervention strategies to minimize ecological and human health damage?",
        ],
    },
    {
        name: "Health",
        description:
            "Tackle bias, transparency, and privacy in medical AI. Design systems that improve drug safety, patient matching, or mental health support while ensuring equitable outcomes.",
        prompts: [
            "While AI can revolutionize diagnostics, black box algorithms often lack the transparency required for clinical accountability and may perpetuate racial or gender biases in treatment recommendations. Develop a system or interface that ensures AI-driven medical decisions are both explainable to practitioners and strictly compliant with patient privacy. How does your solution ensure moral agency in life-altering care?",
            "Develop an AI system that analyzes anonymized health data to detect early warning signs of adverse drug reactions across populations. The platform must minimize false alarms, provide interpretable risk signals to clinicians, and continuously monitor for racial, gender, or age-based bias in its predictions. How can your system improve drug safety while ensuring equitable and accountable decision-making?",
            "Create an AI-powered platform that matches patients to clinical trials while actively correcting for demographic underrepresentation and preventing exploitative recruitment practices. Your system must preserve patient privacy, audit itself for bias, and generate transparent fairness reports for regulators and hospitals. How can AI accelerate pharmaceutical innovation without reinforcing systemic inequities in medical research?",
            "Design a privacy-preserving AI \"clinical memory layer\" that aggregates a patient's diagnoses, medications, lab results, and care history across providers to detect drug interactions, gaps in treatment, and conflicting prescriptions. The system must provide transparent, explainable reasoning while ensuring that patients and physicians retain final decision-making authority. How can your design protect patient autonomy and privacy while safely augmenting clinical judgment?",
            "Medical AI often underperforms on underrepresented populations. This largely derives from biased data sets, model drift, and a lack of data. Develop a solution which addresses these challenges while understanding potential language barriers and cultural differences.",
            "Artificial intelligence in the psychiatric scene has been heavily criticized for its inability to empathize compared to its human counterparts. However, many patients have reflected that it is easier to express their thoughts to a digital program rather than an individual. How can AI driven platforms provide early stage detection of depression and anxiety through speech and behavior patterns without compromising user autonomy and privacy?",
        ],
    },
    {
        name: "Education",
        description:
            "Reimagine AI's role in learning — supporting neurodivergent students, preserving critical thinking, and bridging accessibility gaps in classrooms.",
        prompts: [
            "While AI tutors offer 24/7 support, an over-reliance on machine interaction risks devaluing the social-emotional bond between teacher and student and may overlook the unique needs of neurodivergent learners. Design an AI integration that is specifically tailored to support students with ADHD, autism, or dyslexia without replacing the mentorship of a human educator. How does your solution ensure that AI serves as a tool for deeper social inclusion and personalized emotional support rather than a catalyst for digital isolation?",
            "AI is increasingly shaping what knowledge is prioritized — optimizing for test performance, engagement metrics, and marketable skills. Design a system that audits or rebalances AI-curated curricula to ensure diversity of thought, civic literacy, and long-term intellectual resilience rather than short-term optimization. Who decides what AI decides is worth learning, and how can your system prevent the narrowing of collective knowledge?",
            "As AI tutors become widespread, students may lose independent critical thinking skills. Key issues include overreliance and a lack of metacognitive engagement. Seek to address the aforementioned problems in an engaging manner, keeping in mind privacy concerns and content moderation.",
            "In recent studies, it has been reflected that artificial intelligence tools are neither sufficient nor superior compared to in-person and hands-on learning for K-12 students. How can we design a system that is able to identify learning gaps without directly interfering with the learning process and without reinforcing socioeconomic bias?",
            "While traditional classrooms have advanced significantly in terms of offering personalized and tailored learning plans for students, they often fail when it comes to providing accessible learning alternatives for students with physical disabilities (hearing impairment, vision impairment, etc.). How can we make a traditional classroom more accessible?",
        ],
    },
    {
        name: "Safety",
        description:
            "Build safeguards against AI misuse in high-stakes environments. Address online privacy threats, dual-use risks, and the need for human oversight in critical decisions.",
        prompts: [
            "The deployment of autonomous weapon systems and AI-driven battlefield logistics raises the question of who is responsible when a machine makes a lethal error. Design a protocol or technical safeguard that ensures human control over AI-assisted strategic decisions in high-pressure environments. How can your system prevent automation bias (where commanders over-rely on machine outputs) and maintain adherence to international humanitarian law?",
            "Governments may use AI simulations to model geopolitical outcomes and justify strategic decisions before public scrutiny. Design a transparency or auditing framework that allows civilian oversight bodies to evaluate AI-generated war-game predictions without compromising national security. How can democratic accountability coexist with AI-accelerated strategic decision-making?",
            "Many civilian AI systems can be repurposed militarily. This poses not only domestic safety concerns but also broader ethical questions. Design a system which seeks to flag said dual-use systems while still in development stages.",
            "In recent years AI has drastically decreased the barrier of entry to a variety of malignant activities online, most notably having to do with user privacy and data protection. Design a protocol or technical safeguard that ensures human control over AI-assisted strategic decisions in high-stakes environments.",
        ],
    },
    {
        name: "Labor Market",
        description:
            "Confront AI-driven workforce displacement, hiring bias, and pay gaps. Build tools for ethical reskilling and transparent, fair recruitment practices.",
        prompts: [
            "As AI shifts the value of human labor, we face a reskilling gap that threatens to leave millions behind and concentrate wealth in the hands of a few. Develop a platform or economic mechanism that facilitates proactive, ethical reskilling, matching displaced workers to AI-augmented roles while protecting their livelihood during the transition. How does your solution ensure that the massive productivity gains created by AI lead to shared prosperity rather than a widening wealth gap?",
            "Companies increasingly use AI systems to screen video interviews, analyze speech patterns, and rank candidates — often without disclosing how evaluations are made. Design a system that allows candidates to see, challenge, and contextualize the factors influencing their AI-generated interview scores without exposing proprietary algorithms. How can hiring AI be made contestable and transparent while balancing corporate confidentiality and fairness?",
            "National statistics may mask local labor vulnerability. Build a city-level AI tool that forecasts automation risk and suggests ethical reskilling pathways.",
            "Pay gaps across different ethnicities, genders, or various other forms of identities has been a topic of increased interest since the removal of the DEI Act. How can artificial intelligence systems or other solutions successfully analyze information such as job postings, wage data, and hiring patterns to expose pay gaps and discriminatory practices without violating workers' privacy?",
        ],
    },
    {
        name: "Financial Services",
        description:
            "Fight predatory AI in finance. Create tools that democratize financial literacy, detect discriminatory lending, and ensure equitable access to credit and services.",
        prompts: [
            "While AI can optimize high-frequency trading for the elite, it is often used at the consumer level to power predatory dark patterns or complex debt traps that exploit the financially illiterate. Develop an AI model that deciphers complex contracts in real-time and protects vulnerable users from deceptive lending practices and hyper-personalized scams. How can your solution democratize elite-level wealth management tools for individuals currently underserved or exploited by traditional banking?",
            "AI systems increasingly generate financial news, sentiment signals, and social media narratives that influence retail investor behavior. Design a system that detects AI-generated financial influence campaigns or synthetic market sentiment manipulation. How can markets remain fair when AI can manufacture perception at scale?",
            "AI driven credit scoring models often rely on proxy variables that unintentionally disadvantage marginalized communities. Design a third party auditing tool that detects proxy discrimination and measures disparate impact.",
            "The United States has had a history of discriminatory financial service actions even years after the passing of the Civil Rights Act, purposely redlining historically underprivileged communities and preventing them from accessing economic opportunities. How can we design an AI-powered financial services platform that delivers fast, personalized credit and risk assessment while ensuring fairness, explainability, and equal access for historically underserved communities?",
        ],
    },
]

function TrackModal({ track, onClose }) {
    useEffect(() => {
        const onKey = (e) => e.key === "Escape" && onClose()
        document.addEventListener("keydown", onKey)
        document.body.style.overflow = "hidden"
        return () => {
            document.removeEventListener("keydown", onKey)
            document.body.style.overflow = ""
        }
    }, [onClose])

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            onClick={onClose}
        >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/50" />

            {/* Panel */}
            <div
                className="relative bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-start justify-between p-6 border-b border-gray-100 shrink-0">
                    <div>
                        <p className="text-xs font-light uppercase tracking-widest text-purple-600 mb-1">
                            Challenge Track
                        </p>
                        <h2 className="text-2xl font-medium text-gray-900">{track.name}</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 transition-colors duration-150 ml-4 mt-1"
                        aria-label="Close"
                    >
                        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                            <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                        </svg>
                    </button>
                </div>

                {/* Scrollable body */}
                <div className="overflow-y-auto p-6 space-y-6">
                    <p className="text-gray-500 font-light leading-relaxed">{track.description}</p>
                    <div className="space-y-4">
                        {track.prompts.map((prompt, i) => (
                            <div key={i} className="flex gap-4">
                                <span className="shrink-0 mt-1 w-6 h-6 rounded-full bg-purple-50 text-purple-700 text-xs font-medium flex items-center justify-center">
                                    {i + 1}
                                </span>
                                <p className="text-gray-600 font-light text-sm leading-relaxed">{prompt}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default function HackathonPage() {
    const [isLoaded, setIsLoaded] = useState(false)
    const [selectedTrack, setSelectedTrack] = useState(null)

    useEffect(() => {
        setTimeout(() => window.scrollTo(0, 0), 100)
        setIsLoaded(true)
    }, [])

    return (
        <>
            {selectedTrack && (
                <TrackModal track={selectedTrack} onClose={() => setSelectedTrack(null)} />
            )}

            <div
                className={`transition-all duration-1000 ${
                    isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
                }`}
            >
                {/* Hero */}
                <section className="bg-purple-700 text-white py-24 px-4">
                    <div className="max-w-4xl mx-auto text-center">
                        <p className="text-purple-200 font-light text-base mb-3 uppercase tracking-widest">
                            ShiftSC presents
                        </p>
                        <h1 className="text-5xl md:text-6xl font-medium mb-6">Shift AI Hackathon</h1>
                        <p className="text-xl font-light text-purple-100 mb-12 max-w-2xl mx-auto leading-relaxed">
                            A two-day innovation challenge where technology meets responsibility. Build solutions that
                            shape a more ethical AI future.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-8 justify-center items-center">
                            <div className="text-center">
                                <p className="text-purple-200 text-sm font-light uppercase tracking-wide mb-1">
                                    Day 1
                                </p>
                                <p className="text-white font-medium text-lg">April 11</p>
                                <p className="text-purple-200 font-light text-sm">8:00 PM – 12:00 AM</p>
                            </div>

                            <div className="hidden sm:block w-px h-12 bg-purple-500" />

                            <div className="text-center">
                                <p className="text-purple-200 text-sm font-light uppercase tracking-wide mb-1">
                                    Day 2
                                </p>
                                <p className="text-white font-medium text-lg">April 12</p>
                                <p className="text-purple-200 font-light text-sm">4:00 PM – 10:00 PM</p>
                            </div>

                            <div className="hidden sm:block w-px h-12 bg-purple-500" />

                            <div className="text-center">
                                <p className="text-purple-200 text-sm font-light uppercase tracking-wide mb-1">
                                    Location
                                </p>
                                <p className="text-white font-medium text-lg">THH 450</p>
                                <p className="text-purple-200 font-light text-sm">University of Southern California</p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* About */}
                <section className="py-20 px-4">
                    <div className="max-w-3xl mx-auto text-center">
                        <h2 className="text-3xl font-medium mb-8">About the Event</h2>
                        <p className="text-lg font-light text-gray-600 leading-relaxed mb-6">
                            Join us for the Shift AI Hackathon — a two-day innovation challenge bringing together
                            students, builders, and thinkers to tackle some of the most pressing ethical questions in
                            artificial intelligence.
                        </p>
                        <p className="text-lg font-light text-gray-600 leading-relaxed">
                            Participants will work in{" "}
                            <span className="font-normal text-gray-800">teams of four</span> to address critical
                            challenges across six focus areas. Each team will choose a track and develop thoughtful,
                            impactful solutions that confront real-world ethical concerns in today's rapidly evolving AI
                            landscape. Whether you're passionate about technology, policy, social impact, or innovation
                            — this is your opportunity to build solutions that matter.
                        </p>
                    </div>
                </section>

                {/* Perks */}
                <section className="bg-purple-50 py-20 px-4">
                    <div className="max-w-4xl mx-auto">
                        <h2 className="text-3xl font-medium text-center mb-12">What's Included</h2>
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
                            <div className="bg-white rounded-xl p-6 shadow-sm text-center">
                                <h3 className="text-lg font-medium mb-2">$1,000 in Prizes</h3>
                                <p className="text-gray-500 font-light text-sm leading-relaxed">
                                    Compete for cash prizes across all challenge tracks
                                </p>
                            </div>
                            <div className="bg-white rounded-xl p-6 shadow-sm text-center">
                                <h3 className="text-lg font-medium mb-2">AWS Cloud Credits</h3>
                                <p className="text-gray-500 font-light text-sm leading-relaxed">
                                    Free AWS credits courtesy of the AWS Cloud Club to power your projects
                                </p>
                            </div>
                            <div className="bg-white rounded-xl p-6 shadow-sm text-center">
                                <h3 className="text-lg font-medium mb-2">Industry Speakers</h3>
                                <p className="text-gray-500 font-light text-sm leading-relaxed">
                                    Insights from leaders shaping the future of responsible AI
                                </p>
                            </div>
                            <div className="bg-white rounded-xl p-6 shadow-sm text-center">
                                <h3 className="text-lg font-medium mb-2">Free Food</h3>
                                <p className="text-gray-500 font-light text-sm leading-relaxed">
                                    Meals and snacks provided both days of the event
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Challenge Tracks */}
                <section className="py-20 px-4">
                    <div className="max-w-5xl mx-auto">
                        <h2 className="text-3xl font-medium text-center mb-4">Challenge Tracks</h2>
                        <p className="text-center text-gray-500 font-light mb-12 max-w-2xl mx-auto">
                            Teams choose one or more focus areas and develop solutions that address real-world ethical
                            concerns in AI. Click any track to see the full challenge prompts.
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {tracks.map((track) => (
                                <button
                                    key={track.name}
                                    onClick={() => setSelectedTrack(track)}
                                    className="text-left border border-gray-200 rounded-xl p-6 hover:shadow-lg hover:-translate-y-1 hover:border-purple-200 transition-all duration-300 cursor-pointer group"
                                >
                                    <div className="flex items-start justify-between mb-3">
                                        <h3 className="text-lg font-medium text-purple-700">{track.name}</h3>
                                        <svg
                                            className="shrink-0 ml-2 mt-0.5 text-gray-300 group-hover:text-purple-400 transition-colors duration-200"
                                            width="16" height="16" viewBox="0 0 16 16" fill="none"
                                        >
                                            <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    </div>
                                    <p className="text-gray-500 font-light text-sm leading-relaxed">
                                        {track.description}
                                    </p>
                                    <p className="mt-4 text-xs font-light text-purple-400 group-hover:text-purple-600 transition-colors duration-200">
                                        {track.prompts.length} challenge prompt{track.prompts.length !== 1 ? "s" : ""} →
                                    </p>
                                </button>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Schedule */}
                <section className="bg-gray-50 py-20 px-4">
                    <div className="max-w-2xl mx-auto">
                        <h2 className="text-3xl font-medium text-center mb-12">Schedule</h2>
                        <div className="space-y-5">
                            <div className="bg-white rounded-xl p-6 shadow-sm flex gap-6 items-start">
                                <div className="bg-purple-700 text-white rounded-lg px-4 py-3 text-center min-w-[72px]">
                                    <div className="text-xs font-light uppercase tracking-wide">Apr</div>
                                    <div className="text-2xl font-semibold leading-none mt-1">11</div>
                                </div>
                                <div>
                                    <h3 className="text-lg font-medium mb-1">Day 1 — Kickoff</h3>
                                    <p className="text-gray-400 font-light text-sm mb-2">
                                        8:00 PM – 12:00 AM &middot; THH 450
                                    </p>
                                    <p className="text-gray-600 font-light text-sm leading-relaxed">
                                        Opening ceremony, team formation, challenge reveal, speaker sessions, and hacking
                                        begins.
                                    </p>
                                </div>
                            </div>

                            <div className="bg-white rounded-xl p-6 shadow-sm flex gap-6 items-start">
                                <div className="bg-purple-700 text-white rounded-lg px-4 py-3 text-center min-w-[72px]">
                                    <div className="text-xs font-light uppercase tracking-wide">Apr</div>
                                    <div className="text-2xl font-semibold leading-none mt-1">12</div>
                                </div>
                                <div>
                                    <h3 className="text-lg font-medium mb-1">Day 2 — Finals</h3>
                                    <p className="text-gray-400 font-light text-sm mb-2">
                                        4:00 PM – 10:00 PM &middot; THH 450
                                    </p>
                                    <p className="text-gray-600 font-light text-sm leading-relaxed">
                                        Final project presentations, judging panel, speaker insights, and awards
                                        ceremony.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Sponsors */}
                <section className="py-16 px-4">
                    <div className="max-w-2xl mx-auto text-center">
                        <h2 className="text-2xl font-medium mb-10">Sponsored By</h2>
                        <div className="flex flex-wrap gap-10 justify-center items-center">
                            <span className="text-gray-500 font-light text-base">AWS Cloud Club</span>
                            <span className="text-gray-300 hidden sm:inline">|</span>
                            <span className="text-gray-500 font-light text-base">Alani Energy Drinks</span>
                        </div>
                    </div>
                </section>
            </div>
        </>
    )
}
