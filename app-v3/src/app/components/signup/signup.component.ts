import { Component, inject, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatCardModule } from '@angular/material/card';
import { MatSelectModule } from '@angular/material/select';
import {MatSnackBar, MatSnackBarModule} from '@angular/material/snack-bar';
@Component({
  selector: 'app-signup',
  imports:[CommonModule, ReactiveFormsModule,
    HttpClientModule,
    MatCardModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatSnackBarModule,
  ],
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.css']
})
export class SignupComponent implements OnInit {
  signupForm!: FormGroup;
  errorMessage: string = '';
  successMessage: string = '';
  private _snackBar = inject(MatSnackBar);


  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.signupForm = this.fb.group({
      name: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(8)]],
      phoneNumber: [''], // Optional
    });
  }
  
  onSubmit(): void {
    if (this.signupForm.valid) {
      const formData = {
        name  : this.signupForm.get('name')?.value,
        email: this.signupForm.get('email')?.value,
        password: this.signupForm.get('password')?.value,
        phoneNumber: this.signupForm.get('phoneNumber')?.value || null,
      };

      this.authService.signup(formData).subscribe({
        next: (response) => {
          this.successMessage = 'Registration successful! Please log in.';
          this.errorMessage = '';
          this._snackBar.open('Registration successful! Please log in.','close',  {
            duration: 1000,
          });
          setTimeout(() => this.router.navigate(['/login']), 2000);
        },
        error: (err) => {
          this.errorMessage = err.error.message || 'Error during signup';
          this.successMessage = '';

          this._snackBar.open(err.error.message || 'Error during signup','close',  {
            duration: 1000,
          });
        }
      });
    }
  }
}