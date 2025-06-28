import { Component } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { RouterModule, RouterOutlet } from '@angular/router';
import { MatTableModule } from '@angular/material/table';
import { MatGridListModule } from '@angular/material/grid-list';
import { ApiService } from '../services/api.service';
import { CommonModule } from '@angular/common';
import {MatIconModule} from '@angular/material/icon'
import { MatSnackBar } from '@angular/material/snack-bar';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-dashboard',
  imports: [ RouterModule,  MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatTableModule,
    MatGridListModule,
    FormsModule,
    ReactiveFormsModule,
    MatTableModule,
    CommonModule,
    MatIconModule,

  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent {
  salaryForm: FormGroup;
  expenseForm: FormGroup;
  chatForm: FormGroup;
  salaryResult: any = null;
  expenseResult: any = null;
  predictions: any = null;
  records: any[] = [];
  messages: { query: string, response: string }[] = [];
  displayedColumns: string[] = ['weekly.gross_pay', 'weekly.net_pay', 'monthly.gross', 'yearly.net'];
  private socket: any;

  constructor(
    private fb: FormBuilder,
    private apiService: ApiService,
    private snackBar: MatSnackBar
  ) {
    this.salaryForm = this.fb.group({
      rate: ['', [Validators.required, Validators.min(0)]],
      hours: ['', [Validators.required, Validators.min(0)]]
    });
    this.expenseForm = this.fb.group({
      description: ['', Validators.required],
      amount: ['', [Validators.required, Validators.min(0)]]
    });
    this.chatForm = this.fb.group({
      query: ['', Validators.required]
    });
  }

  ngOnInit() {
    this.loadPredictions();
    this.loadRecords();
    this.initializeSocket();
  }

  initializeSocket() {
    this.socket = io(environment.apiUrl, {
      path: '/socket.io',
      transports: ['websocket'],
      query: { token: this.apiService.getToken() }
    });

    this.socket.on('connect', () => {
      console.log('Connected to Socket.IO server');
      this.socket.emit('join', { user_id: this.apiService.getUserId() });
    });

    this.socket.on('connected', (data: any) => {
      console.log(data.message);
    });

    this.socket.on('joined', (data: any) => {
      console.log(data.message);
    });

    this.socket.on('new_record', (data: any) => {
      if (data.data) {
        if (data.data.weekly && data.data.monthly && data.data.yearly) {
          this.records = [...this.records, data.data];
        } else if (data.data.expenses) {
          this.expenseResult = data.data;
        }
      }
    });

    this.socket.on('notification', (data: any) => {
      this.snackBar.open(data.message, 'Close', { duration: 5000 });
    });

    this.socket.on('chat_message', (data: any) => {
      this.messages = [...this.messages, { query: data.query, response: data.response }];
    });
  }

  loadPredictions() {
    this.apiService.getPredictions().subscribe({
      next: (res) => {
        this.predictions = {
          month: res.month,
          predictions: Object.entries(res.predictions).map(([key, value]) => ({ key, value }))
        };
        this.snackBar.open(res.message, 'Close', { duration: 5000 });
      },
      error: (err) => {
        const errorMessage = err.error?.error || 'Failed to load predictions';
        this.snackBar.open(errorMessage, 'Close', { duration: 5000 });
      }
    });
  }

  loadRecords() {
    this.apiService.getRecords().subscribe({
      next: (res) => {
        this.records = res.filter((record: { weekly: any; monthly: any; yearly: any; }) => record.weekly && record.monthly && record.yearly);
      },
      error: (err) => {
        this.snackBar.open('Failed to load records: ' + err.error?.error, 'Close', { duration: 3000 });
      }
    });
  }

  onSalarySubmit() {
    if (this.salaryForm.invalid) return;
    const { rate, hours } = this.salaryForm.value;
    this.apiService.calculateSalary([{ rate, hours }]).subscribe({
      next: (res) => {
        this.salaryResult = res;
        this.snackBar.open('Salary calculated and saved', 'Close', { duration: 3000 });
      },
      error: (err) => {
        this.snackBar.open('Failed to calculate salary: ' + err.error?.error, 'Close', { duration: 3000 });
      }
    });
  }

  onExpenseSubmit() {
    if (this.expenseForm.invalid) return;
    const { description, amount } = this.expenseForm.value;
    this.apiService.analyzeExpenses([{ description, amount }]).subscribe({
      next: (res) => {
        this.expenseResult = res;
        this.snackBar.open('Expenses analyzed and saved', 'Close', { duration: 3000 });
      },
      error: (err) => {
        this.snackBar.open('Failed to analyze expenses: ' + err.error?.error, 'Close', { duration: 3000 });
      }
    });
  }

  onChatSubmit() {
    if (this.chatForm.invalid) return;
    const query = this.chatForm.value.query;
    this.apiService.chat(query).subscribe({
      next: (res) => {
        this.chatForm.reset();
      },
      error: (err) => {
        this.snackBar.open('Failed to get chat response: ' + err.error?.error, 'Close', { duration: 3000 });
      }
    });
  }

  ngOnDestroy() {
    if (this.socket) {
      this.socket.disconnect();
    }
  }
}
function io(apiUrl: string, arg1: { path: string; transports: string[]; query: { token: string | null; }; }): any {
  throw new Error('Function not implemented.');
}

