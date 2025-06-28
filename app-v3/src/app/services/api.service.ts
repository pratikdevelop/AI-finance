import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from './auth.service';
import { jwtDecode } from "jwt-decode";
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:5000/api';

  constructor(
    private http: HttpClient,
    private authService: AuthService,
  ) {}

  private getHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    return new HttpHeaders({
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`
    });
  }

  getUserId(): any {
    const token = this.authService.getToken();
    if (token) {
      const decoded = jwtDecode(token);
      console.log(
        'dd',
        decoded
      );
      
      return decoded;
    }
    return null;
  }

  getToken(): string | null {
    return this.authService.getToken();
  }

  calculateSalary(rates: any[]): Observable<any> {
    return this.http.post(`${this.apiUrl}/calculate-salary`, { rates }, { headers: this.getHeaders() });
  }

  analyzeExpenses(expenses: any[]): Observable<any> {
    return this.http.post(`${this.apiUrl}/analyze-expenses`, { expenses }, { headers: this.getHeaders() });
  }

  getPredictions(): Observable<any> {
    return this.http.get(`${this.apiUrl}/predict-expenses`, { headers: this.getHeaders() });
  }

  getReports(): Observable<any> {
    return this.http.get(`${this.apiUrl}/reports`, { headers: this.getHeaders() });
  }

  getAnalytics(): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics`, { headers: this.getHeaders() });
  }

  saveData(data: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/save-data`, data, { headers: this.getHeaders() });
  }

  getRecords(): Observable<any> {
    return this.http.get(`${this.apiUrl}/get-records`, { headers: this.getHeaders() });
  }

  chat(query: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/chat`, { query }, { headers: this.getHeaders() });
  }
}