import { useEffect, useRef } from 'react'

interface NeuralVizProps {
  layers: Array<{ name: string; neurons: number }>
  type: 'visual' | 'audio' | 'multimodal'
  animate?: boolean
}

function NeuralViz({ layers, type, animate = true }: NeuralVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>(0)

  const colors = {
    visual: { primary: '#6366f1', secondary: '#818cf8' },
    audio: { primary: '#8b5cf6', secondary: '#a78bfa' },
    multimodal: { primary: '#06b6d4', secondary: '#22d3ee' },
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const { primary, secondary } = colors[type]

    let time = 0

    const draw = () => {
      ctx.clearRect(0, 0, width, height)

      const layerSpacing = width / (layers.length + 1)

      // Draw connections
      ctx.strokeStyle = `${primary}20`
      ctx.lineWidth = 1

      for (let i = 0; i < layers.length - 1; i++) {
        const x1 = layerSpacing * (i + 1)
        const x2 = layerSpacing * (i + 2)
        const n1 = Math.min(layers[i].neurons, 10)
        const n2 = Math.min(layers[i + 1].neurons, 10)

        for (let j = 0; j < n1; j++) {
          const y1 = (height / (n1 + 1)) * (j + 1)
          for (let k = 0; k < n2; k++) {
            const y2 = (height / (n2 + 1)) * (k + 1)

            // Animated connection
            if (animate) {
              const offset = (time * 0.02 + j * 0.1 + k * 0.05) % 1
              const gradient = ctx.createLinearGradient(x1, y1, x2, y2)
              gradient.addColorStop(0, `${primary}00`)
              gradient.addColorStop(offset, `${primary}60`)
              gradient.addColorStop(1, `${primary}00`)
              ctx.strokeStyle = gradient
            }

            ctx.beginPath()
            ctx.moveTo(x1, y1)
            ctx.lineTo(x2, y2)
            ctx.stroke()
          }
        }
      }

      // Draw neurons
      for (let i = 0; i < layers.length; i++) {
        const x = layerSpacing * (i + 1)
        const n = Math.min(layers[i].neurons, 10)

        for (let j = 0; j < n; j++) {
          const y = (height / (n + 1)) * (j + 1)
          const radius = 8

          // Neuron glow
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2)
          gradient.addColorStop(0, primary)
          gradient.addColorStop(1, `${primary}00`)
          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(x, y, radius * 2, 0, Math.PI * 2)
          ctx.fill()

          // Neuron core
          ctx.fillStyle = secondary
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fill()

          // Pulse animation
          if (animate) {
            const pulse = Math.sin(time * 0.05 + j * 0.5) * 0.5 + 0.5
            ctx.strokeStyle = `${primary}${Math.floor(pulse * 255).toString(16).padStart(2, '0')}`
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(x, y, radius + 4 + pulse * 4, 0, Math.PI * 2)
            ctx.stroke()
          }
        }

        // Layer label
        ctx.fillStyle = '#94a3b8'
        ctx.font = '12px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(layers[i].name, x, height - 10)
        ctx.fillText(`(${layers[i].neurons})`, x, height + 5)
      }

      time++
      if (animate) {
        animationRef.current = requestAnimationFrame(draw)
      }
    }

    draw()

    return () => {
      cancelAnimationFrame(animationRef.current)
    }
  }, [layers, type, animate])

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={200}
      className="w-full h-auto"
    />
  )
}

export default NeuralViz
