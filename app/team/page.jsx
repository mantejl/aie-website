"use client"
import Image from "next/image"
import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Linkedin, Users, Award } from "lucide-react"

export default function TeamPage() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    setTimeout(() => {
      window.scrollTo(0, 0)
    }, 100)
    setIsLoaded(true)
  }, [])

  const currentTeam = [
    {
      id: 1,
      name: "Mattice Ureel",
      title: "Initiative Lead",
      imageUrl: "/team/mattice.jpeg",
      linkedin: "https://www.linkedin.com/in/mattice-ureel/",
      isLead: true,
    },
    {
      id: 2,
      name: "Ryan Nene",
      title: "Initiative Lead",
      imageUrl: "/team/ryan.jpeg",
      linkedin: "https://www.linkedin.com/in/ryan-nene-7164b0291/",
      isLead: true,
    },
    {
      id: 3,
      name: "Rida Faraz",
      title: "Core Member",
      imageUrl: "/team/rida.jpeg",
      linkedin: "https://www.linkedin.com/in/rida-faraz/",
    },
    {
      id: 4,
      name: "Richa Misra",
      title: "Core Member",
      imageUrl: "/team/richa.jpeg",
      linkedin: "https://www.linkedin.com/in/richa-misra-m/",
    },
    {
      id: 5,
      name: "Jack Schilson",
      title: "Core Member",
      imageUrl: "/team/jack.jpeg",
      linkedin: "https://www.linkedin.com/in/jack-schilson-9a4213291/",
    },
  ]

  const pastTeam = [
    {
      id: 7,
      name: "Mantej Lamba",
      title: "Former Initiative Lead",
      imageUrl: "/team/mantej.jpeg",
      linkedin: "https://www.linkedin.com/in/mantejlamba/",
    },
    {
      id: 8,
      name: "Ada Basar",
      title: "Former Initiative Lead",
      imageUrl: "/team/ada.jpeg",
      linkedin: "https://www.linkedin.com/in/adanurbasar/",
    },
    {
      id: 9,
      name: "Cole Gawin",
      title: "Former Core Member",
      imageUrl: "/team/cole.jpeg",
      linkedin: "https://www.linkedin.com/in/colegawin/",
    },
    {
      id: 10,
      name: "Samyu Vakkalanka",
      title: "Former Core Member",
      imageUrl: "/team/samyu.jpeg",
      linkedin: "https://www.linkedin.com/in/samyukta-vakkalanka/",
    },
    {
      id: 11,
      name: "Anisha Chitta",
      title: "Former Core Member",
      imageUrl: "/team/anisha.jpeg",
      linkedin: "https://www.linkedin.com/in/anisha-chitta/",
    },
  ]

  const TeamMemberCard = ({ member, size = "normal" }) => (
    <Card
      className={`group hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 bg-gradient-to-br from-white to-gray-50 border-2 ${member.isLead ? "border-purple-300 shadow-lg" : "border-gray-200"} ${size === "large" ? "p-6" : "p-4"}`}
    >
      <div className={`relative ${size === "large" ? "h-[200px] w-[200px]" : "h-[150px] w-[150px]"} mx-auto mb-4`}>
        <div
          className={`relative ${size === "large" ? "h-[200px] w-[200px]" : "h-[150px] w-[150px]"} rounded-full overflow-hidden border-4 ${member.isLead ? "border-purple-400" : "border-gray-300"} group-hover:border-purple-500 transition-colors duration-300`}
        >
          <Image
            src={member.imageUrl || "/placeholder.svg"}
            alt={member.name}
            fill
            className="object-cover group-hover:scale-110 transition-transform duration-300"
          />
        </div>
        {member.isLead && (
          <div className="absolute -top-2 -right-2 bg-purple-600 text-white p-2 rounded-full shadow-lg">
            <Award className="h-4 w-4" />
          </div>
        )}
      </div>
      <CardContent className="text-center p-0">
        <div className="flex items-center justify-center gap-2 mb-2">
          {member.linkedin ? (
            <a
              href={member.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className={`font-bold ${size === "large" ? "text-xl" : "text-lg"} text-gray-800 hover:text-purple-600 transition-colors duration-200 flex items-center gap-2 group-hover:underline`}
            >
              {member.name}
              <Linkedin className="h-4 w-4 text-blue-600" />
            </a>
          ) : (
            <span className={`font-bold ${size === "large" ? "text-xl" : "text-lg"} text-gray-800`}>{member.name}</span>
          )}
        </div>
        <p
          className={`text-gray-600 ${size === "large" ? "text-base" : "text-sm"} font-medium ${member.isLead ? "text-purple-700" : ""}`}
        >
          {member.title}
        </p>
      </CardContent>
    </Card>
  )

  return (
    <div
      className={`min-h-screen transition-all duration-1000 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"}`}
    >
      <div className="container mx-auto px-4 py-12">
        <section className="mb-16 text-center">
          <div className="bg-purple-700 py-12 px-8 rounded-3xl shadow-2xl relative overflow-hidden">
            <div className="absolute inset-0 bg-black/10"></div>
            <div className="relative z-10">
              <div className="flex items-center justify-center gap-3 mb-6">
                <Users className="h-8 w-8 text-white" />
                <h1 className="text-4xl font-medium text-white">Meet the Team</h1>
              </div>

              <div className="flex justify-center mb-6">
                <div className="relative h-[400px] w-[600px] rounded-2xl overflow-hidden shadow-2xl border-4 border-white/30 transition-transform duration-700 hover:scale-105">
                  <Image src="/team/team.png" alt="Team Photo" fill className="object-cover" priority />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
                </div>
              </div>

              <p className="text-sm text-white/90 italic font-light max-w-2xl mx-auto">
                Dedicated to advancing ethical AI together through research, education, and community engagement
              </p>
            </div>
          </div>
        </section>

        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-800 mb-4">Current Team</h2>
            <div className="w-24 h-1 bg-purple-700 mx-auto rounded-full"></div>
          </div>

          <div className="mb-12">
            <h3 className="text-2xl font-semibold text-center mb-8">Initiative Leads</h3>
            <div className="flex justify-center gap-12">
              {currentTeam
                .filter((member) => member.isLead)
                .map((member) => (
                  <TeamMemberCard key={member.id} member={member} size="large" />
                ))}
            </div>
          </div>

          <div>
            <h3 className="text-2xl font-semibold text-center mb-8">Core Members</h3>
            <div className="flex justify-center gap-8">
              {currentTeam
                .filter((member) => !member.isLead)
                .map((member) => (
                  <TeamMemberCard key={member.id} member={member} />
                ))}
            </div>
          </div>
        </section>

        <section>
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-800 mb-4">Past Contributors</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-gray-400 to-gray-600 mx-auto rounded-full"></div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-8 justify-items-center">
            {pastTeam.map((member) => (
              <TeamMemberCard key={member.id} member={member} />
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
