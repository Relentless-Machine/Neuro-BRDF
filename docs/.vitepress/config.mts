import { defineConfig } from 'vitepress'
import markdownItFootnote from 'markdown-it-footnote'

export default defineConfig({
  base: "/Neuro-BRDF/",
  title: "Neuro-BRDF Docs",
  description: "Documentation for Neuro-BRDF reproduction and improvement",
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '复现 (NeRO)', link: '/reproduction/' },
      { text: '改进 (Phases)', link: '/improvement/phase1-plan' },
      { text: '研究 (Research)', link: '/research/1-roadmap' }
    ],

    sidebar: {
      '/reproduction/': [
        {
          text: '基础复现 (Reproduction)',
          items: [
            { text: '项目综述', link: '/reproduction/' },
            { text: '1. 三维表示方法 (SDF)', link: '/reproduction/1-sdf' },
            { text: '2. 着色神经网络 (MLP)', link: '/reproduction/2-mlp' },
            { text: '3. 集成方向编码 (IDE)', link: '/reproduction/3-ide' },
            { text: '4. 训练流程说明', link: '/reproduction/4-training' }
          ]
        }
      ],
      '/improvement/': [
        {
          text: '阶段一：Hash Grid 与软分割 (Phase I)',
          items: [
            { text: '1. 改进方案 (Plan)', link: '/improvement/phase1-plan' },
            { text: '2. 可行性分析 (Feasibility)', link: '/improvement/phase1-feasibility' }
          ]
        },
        {
          text: '阶段二：统一层状模型 (Phase II)',
          items: [
            { text: '3. ULMM 方案', link: '/improvement/phase2-ulmm' }
          ]
        }
      ],
      '/research/': [
        {
          text: '研究与思考 (Research)',
          items: [
            { text: '1. 应用与价值分析 (Roadmap)', link: '/research/1-roadmap' },
            { text: '2. 第一份综述 (Theory)', link: '/research/2-first-summary' },
            { text: '日志 (Log)', link: '/research/log' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Relentless-Machine/Neuro-BRDF' }
    ]
  },
  markdown: {
    math: true,
    config: (md) => {
      md.use(markdownItFootnote)
    }
  }
})
