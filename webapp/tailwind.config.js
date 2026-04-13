/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Deep Space Protocol surface hierarchy
        'surface-lowest':   '#0c0e14',
        'surface-low':      '#191b22',
        'surface':          '#11131a',
        'surface-container':'#1d1f26',
        'surface-high':     '#282a31',
        'surface-highest':  '#33353c',
        'surface-bright':   '#373940',
        'surface-dim':      '#11131a',
        // Accent tokens
        primary:            '#b4c5ff',
        'primary-container':'#2563eb',
        'primary-fixed':    '#dbe1ff',
        secondary:          '#ddb7ff',
        'secondary-container':'#6f00be',
        tertiary:           '#4edea3',
        'tertiary-container':'#007d55',
        // On-surface
        'on-surface':         '#e1e2eb',
        'on-surface-variant': '#c3c6d7',
        'on-primary':         '#002a78',
        // Structure
        outline:          '#8d90a0',
        'outline-variant':'#434655',
        error:            '#ffb4ab',
        'error-container':'#93000a',
      },
      fontFamily: {
        headline: ['"Plus Jakarta Sans"', 'sans-serif'],
        body:     ['Inter', 'sans-serif'],
        label:    ['Inter', 'sans-serif'],
        mono:     ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '0.25rem',
        lg:      '0.5rem',
        xl:      '1rem',
        '2xl':   '1.5rem',
        full:    '9999px',
      },
      backdropBlur: {
        glass: '12px',
      },
      boxShadow: {
        glass: '0 20px 40px rgba(0,0,0,0.4)',
        glow:  '0 0 20px rgba(180,197,255,0.15)',
        'glow-tertiary': '0 0 20px rgba(78,222,163,0.2)',
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4,0,0.6,1) infinite',
        'spin-slow':  'spin 3s linear infinite',
      },
      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
