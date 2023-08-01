
/**
 * weipeng-jiao 2023/07/15
 * note:  std thread 典型例子  
*/
#include <iostream>
#include <thread>
#include <pthread.h>
#include <mutex> // 锁的头文件
#include <condition_variable> // 条件变量的头文件
#include <list>
using namespace std;
using namespace std::literals::chrono_literals; // c++14 特性
std::mutex g_mutex;
condition_variable g_con;
list<int> products;
thread_local int t_l_counter = 0; 


void SayHello() {
    cout << "Hello World" << endl;
}

void SayHelloParam(int id, string name) {
    this_thread::sleep_for(10ms);
    cout << "ID:" << id << ", Hello " << name << endl;
}

// 1. 线程离开
int thread_exit() {
    std::thread t1(SayHello);
    // 等待子线程结束才退出当前线程
    pthread_exit(nullptr);
    //return 0;
}

// 2.线程分离
int thread_detach() {
    // 通过 detach() 函数，将子线程和主线分离，子线程可以独立继续运行，即使主线程结束，子线程也不会结束
    std::thread t1(SayHello);
    t1.detach();
 
    /*线程睡眠*/
    // 让当前线程睡眠 10 毫秒
    this_thread::sleep_for(10ms);
    // 低于C++17使用这行代码  this_thread::sleep_for(chrono::milliseconds(10));
    // 让当前线程睡眠 5 秒
    this_thread::sleep_for(5s);
    return 0;
}


// 3.线程合并
int thread_join() {
    std::thread t1(SayHelloParam, 1, "Wiki"); // 线程传递参数方式
    t1.join();
    return 0;
}


// 4.线程数据loacl
void sub_thread() {
    cout << "flag1 t_l_counter: " << t_l_counter << endl; // 看到的是副线程的全局变量 0
    t_l_counter = 2; // 将副线程全局变量设为2
}

int thread_data_local() {
    // C++11中提供了thread_local，thread_local定义的变量在每个线程都保存一份副本，而且互不干扰，在线程退出的时候自动销毁。
    t_l_counter = 1; // 主线程将主线程的全局变量设为1
    std::thread t1(sub_thread); // 副线程看到的全局变量是副线程的全局变量仍为0
    t1.join();
    cout << "flag2 t_l_counter: " << t_l_counter << endl; // 此全局变量为主线程的 1
    return 0;
}



// 5.互斥锁
void mutex_test() {
    g_mutex.lock(); // 线程资源上锁
    cout << "task start thread ID: " << this_thread::get_id() << endl;
    this_thread::sleep_for(10ms);
    cout << "task end thread ID: " << this_thread::get_id() << endl;
    g_mutex.unlock(); // 线程资源解锁
}

int thread_mutex() {
    std::thread t1(mutex_test);
    std::thread t2(mutex_test);
    std::thread t3(mutex_test);
    t1.join();
    t2.join();
    t3.join();
    return 0;
}


// 5.互斥锁-非阻塞
void mutex_try_lock_test() {
    if(g_mutex.try_lock()) // 线程资源上锁
    {
        cout << "task start thread ID: " << this_thread::get_id() << endl;
        this_thread::sleep_for(10ms);
        cout << "task end thread ID: " << this_thread::get_id() << endl;
        g_mutex.unlock(); // 线程资源解锁
    }
    else
    {
        // do something
        this_thread::sleep_for(5ms);
        cout << "fail to get lock thread ID: " << this_thread::get_id() << endl;
    }

}

int thread_try_lock() {
    std::thread t1(mutex_try_lock_test);
    std::thread t2(mutex_try_lock_test);
    t1.join();
    t2.join();
    return 0;
}


// 6.条件变量：生产者与消费者
void producer() {
    int product_id = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(g_mutex); // 上锁
        products.push_back(++product_id);
        cout << "[producer]生产者 生产: " << product_id << endl;
        lock.unlock(); // 解锁
        // 通知消费者消费
        g_con.notify_one(); // 通知

        if (product_id > 50) {
            break;
        }
        this_thread::sleep_for(2ms);
    }
}
void consumer(){
    while (true) {
        std::unique_lock<std::mutex> lock(g_mutex); // 上锁
        if (products.empty()) {
            cout << "没有产品，等待" << endl;
            // 进入等待，知道有新产品
            g_con.wait(lock); // 线程等待
        } else {
            int product_id = products.front();
            products.pop_front();
            cout << "[consumer]消费者 消费: " << product_id << endl;
            this_thread::sleep_for(2ms);
            if (product_id > 50) break;
        }
    }
}

int thread_condition_variable() {
    // 生产者-消费者模式
    std::thread t1(producer);
    consumer();
    t1.join();
    return 0;
}

class ScopeMutex {
public:
    explicit ScopeMutex(std::mutex &mutex) {
        this->mutex = &mutex;
        this->mutex->lock();
    }

    ~ScopeMutex() {
        this->mutex->unlock();
    }

    std::mutex *mutex;
};

void test() {
    cout << "task prepare thread ID: " << this_thread::get_id() << endl;
    {
        ScopeMutex scopeMutex(g_mutex);
        cout << "task start thread ID: " << this_thread::get_id() << endl;
        this_thread::sleep_for(10ms);
        cout << "task end thread ID: " << this_thread::get_id() << endl;
    }

}



int thread_scopemutex() {
    std::thread t1(test);
    std::thread t2(test);
    std::thread t3(test);
    t1.join();
    t2.join();
    t3.join();
    return 0;
}

int main()
{
    
    int opt=0;
    switch (opt)
    {
        case 0 :
        thread_exit();
        break;
        case 1 :
        thread_detach();
        break;

        case 2 :
        thread_join();
        break;

        case 3 :
        thread_data_local();
        break;

        case 4 :
        thread_mutex();
        break;

        case 5 :
        thread_try_lock();
        break;

        case 6 :
        thread_condition_variable();
        break;

        case 7 :
        thread_scopemutex();
        break;
    
    default:
        break;
    }
    

    return 0;
}