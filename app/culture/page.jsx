"use client"
import Image from "next/image"
import { useState, useEffect, useRef } from "react"
import { Heart } from "lucide-react"

export default function CulturePage() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [visibleImages, setVisibleImages] = useState(new Set())
  const observerRef = useRef(null)

  useEffect(() => {
    setTimeout(() => {
      window.scrollTo(0, 0)
    }, 100)
    setIsLoaded(true)

    // Intersection Observer for scroll animations
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const imageId = entry.target.getAttribute("data-image-id")
            setVisibleImages((prev) => new Set([...prev, imageId]))
          }
        })
      },
      { threshold: 0.1, rootMargin: "50px" },
    )

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect()
      }
    }
  }, [])

  const galleryImages = [
    {
      id: 1,
      src: "/culture/1.jpg",
    },
    {
      id: 2,
      src: "/culture/2.jpg",

    },
    {
      id: 3,
      src: "/culture/3.jpg",

    },
    {
      id: 4,
      src: "/culture/4.jpg",

    },
    {
      id: 5,
      src: "/culture/5.jpg",

    },
    {
      id: 6,
      src: "/culture/6.jpg",

    },
    {
      id: 7,
      src: "/culture/7.jpg",
   
    },
    {
      id: 8,
      src: "/culture/8.jpg",

    },
    {
      id: 9,
      src: "/culture/14.jpg",

    },
    {
      id: 10,
      src: "/culture/9.jpg",

    },
    {
      id: 11,
      src: "/culture/12.jpg",

    },
    {
      id: 12,
      src: "/culture/10.jpg",

    },
    {
      id: 13,
      src: "/culture/15.jpg",

    },
    {
      id: 14,
      src: "/culture/13.jpg",

    },
    {
      id: 15,
      src: "/culture/11.jpg",

    },
    {
      id: 16,
      src: "/culture/16.jpg",
    },
    {
      id: 17,
      src: "/culture/17.jpg",
    },
    {
      id: 18,
      src: "/culture/18.jpg",
    },
    {
      id: 19,
      src: "/culture/19.jpg"
        },
    {
      id: 20,
      src: "/culture/20.jpg",
    },
    {
      id: 21,
      src: "/culture/21.jpg",
    },
    {
      id: 22,
      src: "/culture/22.jpg",
    },
    {
      id: 23,
      src: "/culture/23.jpg",
    },
    {
      id: 24,
      src: "/culture/24.jpg",
    },
    {
      id: 25,
      src: "/culture/25.jpg",
    },
    {
      id: 26,
      src: "/culture/26.jpg",
    },
    {
      id: 27,
      src: "/culture/27.jpg",
    },
    {
      id: 28,
      src: "/culture/28.jpg",
    },
    {
      id: 29,
      src: "/culture/29.jpg",
    },
    {
      id: 30,
      src: "/culture/30.jpg",
    },
    {
      id: 31,
      src: "/culture/31.jpg",
    },
    {
      id: 32,
      src: "/culture/32.jpg",
    },
    {
      id: 33,
      src: "/culture/33.jpg",
    },
    {
      id: 34,
      src: "/culture/34.png",
    },
    {
      id: 35,
      src: "/culture/35.png",
    },
  ]

  useEffect(() => {
    const imageElements = document.querySelectorAll("[data-image-id]")
    imageElements.forEach((el) => {
      if (observerRef.current) {
        observerRef.current.observe(el)
      }
    })

    return () => {
      imageElements.forEach((el) => {
        if (observerRef.current) {
          observerRef.current.unobserve(el)
        }
      })
    }
  }, [])

  return (
    <div
      className={`min-h-screen transition-all duration-1000 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"}`}
    >
      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-purple-700 mt-8 mb-6"></div>
        <div className="relative z-10 container mx-auto px-4 text-center text-white">
          <div className="flex items-center justify-center gap-4 mb-6">
            <h1 className="text-6xl font-medium">Our Culture</h1>
          </div>

          <div className="max-w-4xl mx-auto space-y-6">
            <p className="text-2xl font-light leading-relaxed">
              Our team thrives on collaboration—and that doesn't stop at projects. These are some of the memories we've
              made along the way.
            </p>
            <p className="text-xl font-thin text-white italic">
              Celebrating the friendships and fun that keep us motivated and inspired.
            </p>
          </div>
        </div>
      </section>

      {/* Activities Blurb */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <div className="flex items-center justify-center gap-3 mb-8">
              <h2 className="text-3xl font-bold text-purple-700">Beyond the Work</h2>
            </div>
            <div className="bg-purple-50/60 border border-purple-100 rounded-2xl shadow-lg p-8 max-w-3xl mx-auto">
              <div className="space-y-6 text-xl font-light leading-relaxed">
                <p>🤝 We take pride in not just getting along as teammates, but building real friendships along the way.</p>
                <p>🥤 Embracing our “LA-ness,” we’ll make the occasional trip to Erewhon for overpriced smoothies, or head to In-N-Out for something a little more classic.</p>
                <p>🍽️ Team dinners are a regular thing, along with side quests to explore Rodeo Drive. And there’s almost always snacks at our meetings—because food just makes everything better</p>
                <p>🐾 We’ve spent time visiting an animal shelter, taking a break from our usual routines to unwind, reset, and be around something a little more wholesome. </p>
                <p>🎢 We also spent a full day at Universal Studios, embracing the chaos, screaming on roller coasters, and just having fun as a team.</p>
                <blockquote className="italic text-purple-700 border-l-4 font-extralight border-purple-300 pl-4">
                Good work gets even better when you’re doing it with people you actually enjoy being around.
                </blockquote>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-medium text-gray-800 mb-4">Team Highlights</h2>
            <div className="w-32 h-1 bg-purple-700 mx-auto rounded-full"></div>
          </div>

          {/* 5 Images Per Row Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
            {galleryImages.map((image, index) => (
              <div
                key={image.id}
                data-image-id={image.id}
                className={`group cursor-pointer transition-all duration-700 transform ${
                  visibleImages.has(image.id.toString())
                    ? "opacity-100 translate-y-0 scale-100"
                    : "opacity-0 translate-y-8 scale-95"
                }`}
                style={{
                  transitionDelay: `${(index % 5) * 100}ms`,
                }}
              >
                <div className="relative overflow-hidden rounded-2xl shadow-lg group-hover:shadow-2xl transition-all duration-300">
                  <div className="aspect-square relative">
                    <Image
                      src={image.src || "/placeholder.svg"}
                      alt={image.title}
                      fill
                      className="object-cover group-hover:scale-110 transition-transform duration-500 ease-out"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>    
    </div>
  )
}
