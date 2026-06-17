import { createRouter, createWebHistory } from 'vue-router'
import NeuroViewer from '../views/NeuroViewer.vue'

const routes = [
  {
    path: '/',
    redirect: '/neuro-viewer'
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
