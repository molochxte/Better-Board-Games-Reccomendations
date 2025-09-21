# Neon Database Setup Guide for Langflow

## Quick Setup (Recommended)

### Step 1: Get Your Connection String from Neon Console

1. **Go to Neon Console**: Visit [console.neon.tech](https://console.neon.tech)
2. **Select Your Project**: Click on your existing project
3. **Get Connection Details**:
   - Go to **Dashboard** → **Connection Details**
   - Or go to **Settings** → **Connection String**
4. **Copy Connection String**: 
   - Look for "Connection String" or "URI"
   - It looks like: `postgresql://username:password@ep-xxxxx.us-east-1.aws.neon.tech/dbname?sslmode=require`
   - **Copy this entire string**

### Step 2: Configure the Neon Component in Langflow

1. **Add Neon Database Component** to your flow
2. **Paste Connection String**: 
   - Paste the connection string you copied from Neon Console
   - This should be the **only required field** now
3. **Select Collection**: 
   - Choose from existing tables or create new one
   - Collections are PostgreSQL tables with vector columns
4. **Connect Embedding Model**: 
   - Drag OpenAI Embeddings component
   - Connect it to the Embedding Model input

### Step 3: Test the Connection

1. **Build the Component**: Try to build the vector store
2. **Add Documents**: Connect some documents to ingest
3. **Search**: Test vector search functionality

## What Changed

The component is now **much simpler**:

### ✅ What's Required:
- **Connection String**: Direct connection to your existing Neon database
- **Collection Name**: Choose existing table or create new one
- **Embedding Model**: Connect your embedding model (e.g., OpenAI)

### ❌ What's Removed:
- Complex API key management
- Project ID selection
- Database creation (use your existing database)
- Branch management

## Connection String Format

Your connection string should look like this:
```
postgresql://neondb_owner:your_password@ep-xxxxx.us-east-1.aws.neon.tech/your_database?sslmode=require
```

## Troubleshooting

### If Connection String Doesn't Work:
1. **Check Format**: Make sure it starts with `postgresql://`
2. **Verify Credentials**: Ensure username/password are correct
3. **Test Connection**: Try connecting with a PostgreSQL client first
4. **Check SSL**: Make sure `?sslmode=require` is at the end

### If Collection Creation Fails:
1. **Check Permissions**: Ensure your user can create tables
2. **pgvector Extension**: Make sure pgvector is enabled in your Neon database
3. **Table Names**: Use simple names without spaces or special characters

### Common Collection Names:
- `documents`
- `embeddings`
- `vectors`
- `knowledge_base`

## Example Flow

```
Input Documents → Neon Database Component ← OpenAI Embeddings
                        ↓
                   Search Results
```

## Benefits of This Approach

1. **Simpler**: No complex API management
2. **Faster**: Direct connection to your database
3. **Reliable**: Uses your existing Neon setup
4. **Flexible**: Works with any Neon database you already have

This should resolve all the connection issues you were experiencing!
