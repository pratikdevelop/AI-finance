import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  constructor(private http: HttpClient) {}
  

  signup(userData: any): Observable<any> {
    return this.http.post(`${environment.apiUrl}/api/signup`, userData);
  }
  loginUser(userData: any): Observable<any> {
    return this.http.post(`${environment.apiUrl}/api/login`, userData);
  }
  
  isLoggedIn(): boolean {
    return !!localStorage.getItem('access_token');
  }

  logout(): void {
    localStorage.removeItem('access_token');
  }

  getToken(): string | null {
    return localStorage.getItem('access_token');
  }
}