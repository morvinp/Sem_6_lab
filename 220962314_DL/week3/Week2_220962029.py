#!/usr/bin/env python
# coding: utf-8

# Week 2 Lab
# 

# In[8]:


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.device_count() , device



# In[ ]:


#q1


# In[9]:


a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = 2*a + 3*b
y = 5*a**2 + 3*b**3
z = 2*x + 3*y

z.backward() 

print(f"Gradient of z with respect to a: {a.grad.item()}")



# In[ ]:


#q2


# In[10]:


x = torch.tensor(1.0, requires_grad=True)  
b = torch.tensor(1.0, requires_grad=True)  
w = torch.tensor(1.0, requires_grad=True)  

u = w * x
v = u + b
a = torch.relu(v)

a.backward()

print(f"Gradient of a with respect to w: {w.grad.item()}")


# In[ ]:


#q3


# In[11]:


x = torch.tensor(1.0, requires_grad=True)  
b = torch.tensor(1.0, requires_grad=True)  
w = torch.tensor(1.0, requires_grad=True) 

u = w * x
v = u + b
a = torch.sigmoid(v)

a.backward()

print(f"Gradient of a with respect to w: {w.grad.item()}")


# In[ ]:


#q4


# In[12]:


x = torch.tensor(1.0, requires_grad=True)

u = x**2
v = 2*x
w = torch.sin(x)
f = torch.exp(-u - v - w)

f.backward()

print(f"Gradient of f with respect to x: {x.grad.item()}")


# In[ ]:


#q5


# In[13]:


x=torch.tensor(2.0, requires_grad=True)

y=8*x**4+3*x**3+7*x**2+6*x+3
y.backward()

print(f"Gradient of y with respect to x: {x.grad.item()}")


# In[ ]:


#q6


# In[14]:


x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

b = torch.sin(y)
print(f"b (sin(y)): {b.item()}")

a = 2 * x
print(f"a (2 * x): {a.item()}")

c = a / b
print(f"c (a / b): {c.item()}")

d = c * z
print(f"d (c * z): {d.item()}")

e = torch.log(d + 1)
print(f"e (log(d + 1)): {e.item()}")

f = torch.tanh(e)
print(f"f (tanh(e)): {f.item()}")

f.backward()


print(f"Gradient of f with respect to y: {y.grad.item()}")


# In[ ]:




