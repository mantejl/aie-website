"use client"
import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { useEffect, useState } from "react"

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    // Add a small delay to ensure the page is rendered
    setTimeout(() => {
      window.scrollTo(0, 0)
    }, 100)
    
    setIsLoaded(true)
  }, [])

  return (
    <div className={`container mx-auto px-4 py-8 transition-all duration-1000 ${
      isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
    }`}>
      <section className="py-12 flex flex-col md:flex-row gap-8 items-center">
        <div className="flex-1">
          <h1 className="text-5xl font-normal mb-2">AI Ethics at ShiftSC</h1>
          <p className="text-xl font-light mb-6">Shaping the future of responsible AI</p>
          <Button asChild className="font-light bg-purple-700 hover:bg-purple-800">
            <Link href="/work">View Our Work</Link>
          </Button>
        </div>
        <div className="flex-1">
          <div className="relative h-[400px] w-full">
            <Image
              src="/connect-team.jpeg"
              alt="AI Ethics Team"
              fill
              className="object-cover rounded-lg"
              style = {{objectPosition: "center 45%"}}
              priority
            />
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-12 grid md:grid-cols-2 gap-8 items-center">
        <div className="bg-purple-700 text-white p-8 rounded-lg">
          <h2 className="text-3xl font-normal mb-6">Our Mission</h2>
          <div className="relative h-[300px] w-full mb-6">
            <Image
              src="/mission.JPG?height=200&width=400"
              alt="Our Mission"
              fill
              className="object-cover rounded-lg"
            />
          </div>
        </div>
        <div>
          <p className="text-lg mb-4">
            The AI Ethics initiative (AIE) is one of the founding programs at Shift SC—University of Southern
            California's only student-run organization dedicated to advancing humane and ethically grounded technology.
          </p>
          <p className="text-lg mb-4">
            AIE was created to explore the complex and evolving role of artificial intelligence in society through
            research, education, and community collaboration. Our team has led projects across USC, the greater Los
            Angeles community, and on digital platforms, working to spark open conversations and drive meaningful
            change.
          </p>
          <p className="text-lg">
            Whether through workshops, presentations, or public outreach, AIE is committed to shaping the future
            of responsible AI that is inclusive, thoughtful, and centered on the public good.
          </p>
        </div>
      </section>

      {/* Companies Section */}
      <section className="py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-light mb-4">Member Career Highlights</h2>
        </div>
        
        <div className="grid grid-cols-5 gap-8 gap-y-16 items-center justify-items-center max-w-4xl mx-auto">
          <div className="w-24 h-24 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/Amazon Logo 512.webp"
              alt="Amazon"
              width={96}
              height={96}
              className="object-cover w-full h-full"
            />
          </div>

          <a href="https://www.nuna.com" target="_blank" rel="noopener noreferrer">
            <div className="w-24 h-24 rounded-full flex items-center justify-center overflow-hidden cursor-pointer">
              <Image
                src="/company/Nuna Logo.jpg"
                alt="Nuna"
                width={96}
                height={96}
                className="object-cover w-full h-full"
              />
            </div>
          </a>

          <div className="w-24 h-24 bg-gray-200 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/Beats by Dre Logo.svg"
              alt="Beats by Dre"
              width={96}
              height={96}
              className="object-cover w-full h-full"
            />
          </div>

          <div className="w-24 h-24 bg-gray-200 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/LA District Attorney Logo.png"
              alt="LA District Attorney"
              width={96}
              height={96}
              className="object-cover w-full h-full"
            />
          </div>

          <div className="w-32 h-32 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/49ers Logo Circle.png"
              alt="San Francisco 49ers"
              width={96}
              height={96}
              className="object-contain w-full h-full p-2"
            />
          </div>


          <div className="w-28 h-28 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/Stanford Logo.webp"
              alt="Stanford"
              width={96}
              height={96}
              className="object-cover w-full h-full"
            />
          </div>

          <div className="w-32 h-32 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/Michigan.png"
              alt="Michigan"
              width={96}
              height={96}
              className="object-contain w-24 h-24"
            />
          </div>

          <div className="w-32 h-32 rounded-full flex items-center justify-center overflow-hidden">
            <Image
              src="/company/Comcast Logo Icon.png"
              alt="Comcast"
              width={96}
              height={96}
              className="object-contain w-full h-full p-1"
            />
          </div>

          <a href="https://www.atlassian.com" target="_blank" rel="noopener noreferrer">
            <div className="w-36 h-36 pb-8 rounded-full flex items-center justify-center overflow-hidden">
              <Image
                src="/company/Atlassian Logo.svg"
                alt="Atlassian"
                width={96}
                height={96}
                className="object-contain w-full h-full"
              />
            </div>
          </a>

          <a href="https://withtandem.com/" target="_blank" rel="noopener noreferrer">
            <div className="w-24 h-24 bg-gray-200 rounded-full flex items-center justify-center overflow-hidden cursor-pointer">
              <Image
                src="/company/Tandem Logo.jpeg"
                alt="Tandem"
                width={96}
                height={96}
                className="object-cover w-full h-full"
              />
            </div>
          </a>

        </div>
      </section>
    </div>
  )
}
