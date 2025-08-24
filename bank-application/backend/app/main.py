from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import auth, customers, transactions, dashboard , accounts , payments
from app.config.database import engine
from app.db.base import Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bank Management System", version="1.0.0")

# CORS middleware - Critical for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register ALL routers - Make sure customers is included
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(customers.router, prefix="/api/v1/customers", tags=["Customers"])  # This was missing
app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["Transactions"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(accounts.router, prefix="/api/v1/accounts", tags=["Accounts"])
app.include_router(payments.router, prefix="/api/v1/payments", tags=["Payments"])  # Add this line

@app.get("/")
def root():
    return {"message": "Bank Management System API"}
