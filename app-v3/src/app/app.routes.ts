import { Routes } from '@angular/router';
import { SignupComponent } from './components/signup/signup.component';
import { LoginComponent } from './components/login/login.component';

export const routes: Routes = [
  { path: 'auth/signup', 
    loadComponent: () => import('./components/signup/signup.component').then((m) => m.SignupComponent)
   },
  {
    path: 'dashboard', loadComponent: () => import('./dashboard/dashboard.component').then((m) => m.DashboardComponent),
  },
  {
    path: "auth/login",
    loadComponent: () => import('./components/login/login.component').then((m) => m.LoginComponent)
  },
  {
    path:'reports',
    loadComponent:() => import('./reports/reports.component').then(m => m.ReportsComponent)
  },
  {
    path:"analytics",
    loadComponent: () => import('./analytics/analytics.component').then(m=> m.AnalyticsComponent)
  },
  {
    path:"",
    loadComponent: () => import('./landing/landing.component').then(m=>m.LandingComponent)
  }
];
