"use client"
import { useState, useEffect } from "react"

const tracks = [
    {
        name: "Sustainability",
        description:
            "Address AI's growing environmental footprint — from e-waste and energy consumption to freshwater scarcity. Develop frameworks that make AI computation more sustainable.",
    },
    {
        name: "Health",
        description:
            "Tackle bias, transparency, and privacy in medical AI. Design systems that improve drug safety, patient matching, or mental health support while ensuring equitable outcomes.",
    },
    {
        name: "Education",
        description:
            "Reimagine AI's role in learning — supporting neurodivergent students, preserving critical thinking, and bridging accessibility gaps in classrooms.",
    },
    {
        name: "Safety",
        description:
            "Build safeguards against AI misuse in high-stakes environments. Address online privacy threats, dual-use risks, and the need for human oversight in critical decisions.",
    },
    {
        name: "Labor Market",
        description:
            "Confront AI-driven workforce displacement, hiring bias, and pay gaps. Build tools for ethical reskilling and transparent, fair recruitment practices.",
    },
    {
        name: "Financial Services",
        description:
            "Fight predatory AI in finance. Create tools that democratize financial literacy, detect discriminatory lending, and ensure equitable access to credit and services.",
    },
]

export default function HackathonPage() {
    const [isLoaded, setIsLoaded] = useState(false)

    useEffect(() => {
        setTimeout(() => window.scrollTo(0, 0), 100)
        setIsLoaded(true)
    }, [])

    return (
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
                        concerns in AI.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {tracks.map((track) => (
                            <div
                                key={track.name}
                                className="border border-gray-200 rounded-xl p-6 hover:shadow-lg hover:-translate-y-1 transition-all duration-300"
                            >
                                <h3 className="text-lg font-medium mb-3 text-purple-700">{track.name}</h3>
                                <p className="text-gray-500 font-light text-sm leading-relaxed">
                                    {track.description}
                                </p>
                            </div>
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
    )
}
