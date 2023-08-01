#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
using namespace std;

class Thread
{
public:
    Thread();
    virtual ~Thread();

    enum State
    {
        Stoped,     ///<停止状态，包括从未启动过和启动后被停止
        Running,    ///<运行状态
        Paused      ///<暂停状态
    };

    State state() const;

    void start();
    void stop();
    void pause();
    void resume();

protected:
    virtual void process() = 0;

private:
    void run();

private:
    std::thread* _thread;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::atomic_bool _pauseFlag;   ///<暂停标识
    std::atomic_bool _stopFlag;   ///<停止标识
    State _state;
};

hread::Thread()
    : _thread(nullptr),
      _pauseFlag(false),
      _stopFlag(false),
      _state(Stoped)
{

}

Thread::~Thread()
{
    stop();
}

Thread::State Thread::state() const
{
    return _state;
}

void Thread::start()
{
    if (_thread == nullptr)
    {
        _thread = new thread(&Thread::run, this);
        _pauseFlag = false;
        _stopFlag = false;
        _state = Running;
    }
}

void Thread::stop()
{
    if (_thread != nullptr)
    {
        _pauseFlag = false;
        _stopFlag = true;
        _condition.notify_all();  // Notify one waiting thread, if there is one.
        _thread->join(); // wait for thread finished
        delete _thread;
        _thread = nullptr;
        _state = Stoped;
    }
}

void Thread::pause()
{
    if (_thread != nullptr)
    {
        _pauseFlag = true;
        _state = Paused;
    }
}

void Thread::resume()
{
    if (_thread != nullptr)
    {
        _pauseFlag = false;
        _condition.notify_all();
        _state = Running;
    }
}

void Thread::run()
{
    cout << "enter thread:" << this_thread::get_id() << endl;

    while (!_stopFlag)
    {
        process();
        if (_pauseFlag)
        {
            unique_lock<mutex> locker(_mutex);
            while (_pauseFlag)
            {
                _condition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();
        }
    }
    _pauseFlag = false;
    _stopFlag = false;

    cout << "exit thread:" << this_thread::get_id() << endl;
}


void mySleep(int s)
{
    std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

class MyThread : public Thread
{
protected:
    virtual void process() override
    {
        cout << "do my something" << endl;
        mySleep(1);
    }
};

int main(int argc, char *argv[])
{

    MyThread thread;

    cout << "start thread" << endl;
    thread.start();
    cout << "thread state:" << thread.state() << endl;
    mySleep(3);

    cout << "pause thread" << endl;
    thread.pause();
    cout << "thread state:" << thread.state() << endl;
    mySleep(3);

    cout << "resume thread" << endl;
    thread.resume();
    cout << "thread state:" << thread.state() << endl;
    mySleep(3);

    cout << "stop thread" << endl;
    thread.stop();
    cout << "thread state:" << thread.state() << endl;
    mySleep(3);

}