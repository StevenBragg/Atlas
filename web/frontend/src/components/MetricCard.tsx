import { ReactNode } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon?: ReactNode
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  color?: 'primary' | 'secondary' | 'accent' | 'success' | 'warning' | 'error'
}

const colorClasses = {
  primary: 'text-atlas-primary bg-atlas-primary/10 border-atlas-primary/20',
  secondary: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
  accent: 'text-cyan-400 bg-cyan-400/10 border-cyan-400/20',
  success: 'text-green-400 bg-green-400/10 border-green-400/20',
  warning: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20',
  error: 'text-red-400 bg-red-400/10 border-red-400/20',
}

function MetricCard({
  title,
  value,
  subtitle,
  icon,
  trend,
  trendValue,
  color = 'primary',
}: MetricCardProps) {
  return (
    <div className="glass rounded-xl p-6 card-hover">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400 mb-1">{title}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
          {subtitle && <p className="text-sm text-slate-500 mt-1">{subtitle}</p>}
        </div>
        {icon && (
          <div className={`p-3 rounded-lg border ${colorClasses[color]}`}>
            {icon}
          </div>
        )}
      </div>

      {trend && trendValue && (
        <div className="mt-4 flex items-center gap-2">
          {trend === 'up' && <TrendingUp className="w-4 h-4 text-green-400" />}
          {trend === 'down' && <TrendingDown className="w-4 h-4 text-red-400" />}
          {trend === 'neutral' && <Minus className="w-4 h-4 text-slate-400" />}
          <span
            className={`text-sm ${
              trend === 'up'
                ? 'text-green-400'
                : trend === 'down'
                ? 'text-red-400'
                : 'text-slate-400'
            }`}
          >
            {trendValue}
          </span>
        </div>
      )}
    </div>
  )
}

export default MetricCard
