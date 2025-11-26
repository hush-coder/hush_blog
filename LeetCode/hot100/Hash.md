# C++ STL 哈希表详解

C++ STL中的哈希表是基于哈希函数实现的数据结构，提供了平均O(1)的查找、插入和删除操作。C++11引入了四个主要的哈希容器：

- `std::unordered_map` - 键值对映射
- `std::unordered_set` - 唯一元素集合  
- `std::unordered_multimap` - 允许重复键的映射
- `std::unordered_multiset` - 允许重复元素的集合

## unordered_map：键值对映射

### 基本声明

```cpp
// 基本声明
unordered_map<string, int> mp;
// 这里的string是键，int是值。

// 初始化
unordered_map<string, int> mp = {{"apple", 5}, {"banana", 3}};
```

### 常用操作

1. 插入元素

```cpp
mp["orange"] = 8;           // 方式1
mp.insert({"grape", 4});    // 方式2

// 插入多个元素
mp.insert({{"a", 1}, {"b", 2}, {"c", 3}});

// 使用emplace（更高效）
mp.emplace("key", 42);
```

> 注意这里的`mp["orange"]`是该键对应的值的引用

2. 查找元素

```cpp
// 查找元素
if (mp.find("apple") != mp.end()) {
    cout << "找到了" << endl;
}

// 安全访问（推荐）
if (mp.contains("key")) {        // C++20
    int value = mp["key"];
}
```

`find()`返回的是**迭代器**：成功找到返回该键值对的迭代器，未找到则返回end()迭代器。

3. 获取值

```cpp
// 获取值
int count = mp["apple"];    // 如果不存在会创建并返回0
int count2 = mp.at("apple"); // 如果不存在会抛出异常
```

`at()`方法返回的是**对应值的引用（int&）**，不存在的时候就**抛出`std::out_of_range`异常**

`[]`方法返回的同样是**值的引用**，但是当键不存在的时候会**创建一个会创建 `{***, 0}`，并返回 0**

4. 删除元素

```cpp
// 删除元素
mp.erase("apple");
```

5. 其它

```cpp
mp.size();        // 元素个数
mp.empty();       // 是否为空
mp.clear();       // 清空
mp.count("key");  // 返回1或0（因为键唯一）
```

## unordered_set：基于哈希表的集合容器

```cpp
st.insert(5);                 // 插入（已存在则无变化）
st.emplace(7);                // 原地构造插入
st.erase(5);                  // 按键删除，返回删除个数
bool has = st.count(7);       // 是否存在（0/1）
auto it = st.find(7);         // 找到返回迭代器，否则 end()
st.size(); st.empty(); st.clear();
```

> 与unordered_map的区别：unordered_set是存唯一键的。

