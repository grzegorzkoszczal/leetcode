# Cookbook for using PostgreSQL on Linux

## Installation

### Ubuntu

Tutorial:\
https://www.postgresql.org/docs/current/install-make.html

### Arch Linux

```
sudo pacman -Syu
sudo pacman -S postgresql
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -u postgres createuser --interactive
sudo -u postgres createdb <database_name>
sudo -u postgres psql
```

## Basic concepts

### Storing data

Data is stored in the tables.
Each table is a named collections of rows.
ach row of a given table has the same set of named columns, and each column is of a specific data type.
Tables are grouped into databases.
Collection of databases managed by a single PostgreSQL server instance constitutes a database cluster.

### Commands

Change user to postgres in order to work on database

```
sudo su postgres
```

Creating database:

```
createdb <database_name>
```

Deleting database:

```
dropdb <database_name>
```

Start working on the database:

```
psql <database_name>
```