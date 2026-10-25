import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'raman': {
          50: '#eef1fd',
          100: '#dfe4fb',
          200: '#c3cbf6',
          300: '#97a5ee',
          400: '#6476e4',
          500: '#2f49d9',
          600: '#2638b5',
          700: '#1f2e8f',
          800: '#1a2670',
          900: '#141c4f',
        },
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
