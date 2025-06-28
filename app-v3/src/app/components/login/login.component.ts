import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Component, inject } from '@angular/core';
import { ReactiveFormsModule, FormGroup, FormBuilder, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  imports: [CommonModule, ReactiveFormsModule,
    HttpClientModule,
    MatCardModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatSnackBarModule,],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css'
})
export class LoginComponent {
  signupForm!: FormGroup;
  private _snackBar = inject(MatSnackBar);


  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.signupForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(8)]],
    });
  }

  onSubmit(): void {
    if (this.signupForm.valid) {
      const formData = {
        email: this.signupForm.get('email')?.value,
        password: this.signupForm.get('password')?.value,
      };

      this.authService.loginUser(formData).subscribe({
        next: (response) => {
          localStorage.setItem('accessToken', response.access_token);
          // localStorage.setItem('refreshToken', response.refreshToken);
          this._snackBar.open('Login successfully !','close',  {
            duration: 1000,
          });
          this.router.navigate(['/dashboard']);
        },
        error: (err) => {

          this._snackBar.open(err.error.message || 'Error during signup','close',  {
            duration: 1000,
          });
        }
      });
    }
  }
}
