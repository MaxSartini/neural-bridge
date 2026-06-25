import { createRouter, createWebHistory } from 'vue-router'
import EvidenceDashboard from '../views/EvidenceDashboard.vue'
import NeuroViewer from '../views/NeuroViewer.vue'

const routes = [
  {
    path: '/',
    redirect: '/evidence-dashboard'
  },
  {
    path: '/evidence-dashboard',
    name: 'EvidenceDashboard',
    component: EvidenceDashboard
  },
  {
    path: '/neuro-viewer',
    name: 'NeuroViewer',
    component: NeuroViewer
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
