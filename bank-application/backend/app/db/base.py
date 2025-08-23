from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# DO NOT import models here - this causes circular imports
# Models should import Base, not the other way around
